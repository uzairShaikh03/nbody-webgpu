import { gaussian } from './math';

export type PresetId = 'galaxy' | 'collision' | 'sphere' | 'solar';

/** Typed arrays handed to WebGPU must be backed by a plain ArrayBuffer. */
export type BodyArray = Float32Array<ArrayBuffer>;

export interface BodySet {
  /** xyz = position, w = mass, 4 floats per body. */
  positions: BodyArray;
  /** xyz = velocity, w = |acceleration| (filled by the solver). */
  velocities: BodyArray;
  count: number;
  /** Approximate world-space radius of the system, used to frame the camera. */
  extent: number;
  /** Typical speed in the system; normalizes the velocity colour ramp. */
  referenceSpeed: number;
  /** Suggested integrator timestep for this scenario. */
  suggestedDt: number;
  label: string;
}

/**
 * Circular-orbit speed at radius r inside enclosed mass m, in a Plummer-softened
 * potential: v² = G m r² / (r² + eps²)^{3/2}. With G = 1 in simulation units, the
 * total system mass is what sets the clock.
 *
 * Matching the solver's softening matters. The unsoftened v = sqrt(G m / r)
 * diverges at small r, so seeding with it hands bodies near the core speeds the
 * integrator cannot resolve, and they get ejected on the first step.
 */
function circularSpeed(enclosedMass: number, radius: number, softening = 0): number {
  const r2 = radius * radius;
  const denom = Math.pow(r2 + softening * softening, 1.5);
  return radius * Math.sqrt(enclosedMass / Math.max(denom, 1e-9));
}

/** Softening the presets assume; matches the solver default. */
const SEED_SOFTENING = 0.35;

interface DiskOptions {
  count: number;
  radius: number;
  diskMass: number;
  coreMass: number;
  thickness: number;
  /**
   * Radial concentration exponent: r ~ u^concentration for uniform u. Values
   * above 1 pull bodies toward the centre; below 1 pushes them into a ring,
   * because u^p > u for p < 1.
   */
  concentration: number;
  /**
   * Random velocity as a fraction of the local orbital speed. This is what sets
   * the Toomre stability of the disk: too cold and self-gravity collapses it
   * into discrete clumps, too hot and it puffs into a featureless blob.
   */
  dispersion: number;
  center: [number, number, number];
  bulkVelocity: [number, number, number];
  /** Spin axis tilt in radians, so colliding disks are not coplanar. */
  tilt: number;
  spin: 1 | -1;
}

/**
 * Seeds a rotationally supported disk galaxy: one heavy central body plus a
 * flattened distribution on near-circular orbits.
 *
 * Radii are drawn as r = rMin + (R - rMin) * u^concentration, concentrating mass
 * toward the centre while keeping a hole around the core. Because r is monotonic
 * in the uniform sample u, the fraction of bodies inside r is exactly u, so
 * enclosed mass is just core + disk * u -- which is what lets every body start
 * on a balanced orbit instead of collapsing or flying apart.
 */
function seedDisk(
  positions: Float32Array,
  velocities: Float32Array,
  offset: number,
  opts: DiskOptions,
): void {
  const { count, radius, diskMass, coreMass, thickness, concentration, dispersion, center, bulkVelocity, tilt, spin } =
    opts;
  const bodyMass = diskMass / Math.max(count - 1, 1);
  // Inner hole: nothing orbits so close to the core that its orbital period is
  // shorter than a handful of timesteps.
  const rMin = radius * 0.085;

  const cosT = Math.cos(tilt);
  const sinT = Math.sin(tilt);

  // Central body.
  positions.set([center[0], center[1], center[2], coreMass], offset * 4);
  velocities.set([bulkVelocity[0], bulkVelocity[1], bulkVelocity[2], 0], offset * 4);

  for (let i = 1; i < count; i++) {
    const u = Math.random();
    const r = rMin + (radius - rMin) * Math.pow(u, concentration);
    const theta = Math.random() * Math.PI * 2;

    // Disk-local frame: x-z plane is the disk, y is the spin axis.
    const lx = r * Math.cos(theta);
    const lz = r * Math.sin(theta);
    const ly = gaussian() * thickness * (1 - 0.6 * (r / radius));

    // r is monotonic in u, so exactly a fraction u of the disk lies inside r.
    const enclosed = coreMass + diskMass * u;
    const vc = circularSpeed(enclosed, r, SEED_SOFTENING) * spin;

    // Tangential direction in the disk plane.
    let vx = -Math.sin(theta) * vc + gaussian() * vc * dispersion;
    let vz = Math.cos(theta) * vc + gaussian() * vc * dispersion;
    let vy = gaussian() * vc * dispersion * 0.5;

    // Tilt the disk about the x axis.
    const py = ly * cosT - lz * sinT;
    const pz = ly * sinT + lz * cosT;
    const tvy = vy * cosT - vz * sinT;
    const tvz = vy * sinT + vz * cosT;
    vy = tvy;
    vz = tvz;

    const o = (offset + i) * 4;
    positions[o + 0] = center[0] + lx;
    positions[o + 1] = center[1] + py;
    positions[o + 2] = center[2] + pz;
    positions[o + 3] = bodyMass;
    velocities[o + 0] = bulkVelocity[0] + vx;
    velocities[o + 1] = bulkVelocity[1] + vy;
    velocities[o + 2] = bulkVelocity[2] + vz;
    velocities[o + 3] = 0;
  }
}

function allocate(count: number): { positions: BodyArray; velocities: BodyArray } {
  return {
    positions: new Float32Array(count * 4),
    velocities: new Float32Array(count * 4),
  };
}

function galaxy(count: number): BodySet {
  const { positions, velocities } = allocate(count);
  // Bulge-dominated, like a real disk galaxy inside its dark matter halo. A
  // disk that outweighs its centre is Toomre-unstable and fragments into
  // discrete clumps instead of holding a smooth, shearing spiral pattern.
  const diskMass = 7000;
  const coreMass = 18000;
  const radius = 48;

  seedDisk(positions, velocities, 0, {
    count,
    radius,
    diskMass,
    coreMass,
    thickness: 1.0,
    concentration: 1.5,
    dispersion: 0.15,
    center: [0, 0, 0],
    bulkVelocity: [0, 0, 0],
    tilt: 0,
    spin: 1,
  });

  return {
    positions,
    velocities,
    count,
    extent: radius * 1.45,
    referenceSpeed: circularSpeed(coreMass + diskMass, radius * 0.25, SEED_SOFTENING),
    suggestedDt: 0.005,
    label: 'spiral disk galaxy',
  };
}

function collision(count: number): BodySet {
  const { positions, velocities } = allocate(count);
  const half = Math.floor(count / 2);
  const radius = 34;
  const diskMass = 4000;
  const coreMass = 10000;
  // Roughly the speed needed for a grazing, bound encounter rather than a flyby.
  const approach = 0.42 * circularSpeed(2 * (diskMass + coreMass), 110);

  seedDisk(positions, velocities, 0, {
    count: half,
    radius,
    diskMass,
    coreMass,
    thickness: 1.0,
    concentration: 1.5,
    dispersion: 0.13,
    center: [-58, 0, -14],
    bulkVelocity: [approach, 0, approach * 0.28],
    tilt: 0.22,
    spin: 1,
  });

  seedDisk(positions, velocities, half, {
    count: count - half,
    radius: radius * 0.88,
    diskMass: diskMass * 0.8,
    coreMass: coreMass * 0.8,
    thickness: 1.0,
    concentration: 1.5,
    dispersion: 0.13,
    center: [58, 0, 14],
    bulkVelocity: [-approach, 0, -approach * 0.28],
    tilt: -0.9,
    spin: -1,
  });

  return {
    positions,
    velocities,
    count,
    extent: 150,
    referenceSpeed: circularSpeed(coreMass + diskMass, radius * 0.3, SEED_SOFTENING),
    suggestedDt: 0.005,
    label: 'two-galaxy collision',
  };
}

function sphere(count: number): BodySet {
  const { positions, velocities } = allocate(count);
  const radius = 42;
  const totalMass = 16000;
  const bodyMass = totalMass / count;

  for (let i = 0; i < count; i++) {
    // Uniform inside a sphere: direction from a normalized gaussian triple,
    // radius from the cube root of a uniform sample.
    const dir = [gaussian(), gaussian(), gaussian()];
    const len = Math.hypot(dir[0], dir[1], dir[2]) || 1;
    const r = radius * Math.cbrt(Math.random());

    const o = i * 4;
    positions[o + 0] = (dir[0] / len) * r;
    positions[o + 1] = (dir[1] / len) * r;
    positions[o + 2] = (dir[2] / len) * r;
    positions[o + 3] = bodyMass;

    // "Cold" start: almost no kinetic energy, so the cloud free-falls, overshoots
    // through its own centre, and violently relaxes into a core plus halo.
    const vScale = circularSpeed(totalMass, radius) * 0.06;
    velocities[o + 0] = gaussian() * vScale;
    velocities[o + 1] = gaussian() * vScale;
    velocities[o + 2] = gaussian() * vScale;
    velocities[o + 3] = 0;
  }

  return {
    positions,
    velocities,
    count,
    extent: radius * 1.5,
    referenceSpeed: circularSpeed(totalMass, radius * 0.2),
    suggestedDt: 0.003,
    label: 'cold collapse sphere',
  };
}

/** Sun + 8 planets on circular orbits, plus asteroid and Kuiper belts for structure. */
function solar(requested: number): BodySet {
  const sunMass = 9000;
  const planets: Array<{ r: number; m: number }> = [
    { r: 7, m: 0.5 },
    { r: 10.5, m: 1.2 },
    { r: 14, m: 1.5 },
    { r: 19, m: 0.3 },
    { r: 34, m: 40 },
    { r: 48, m: 12 },
    { r: 62, m: 2 },
    { r: 74, m: 2.4 },
  ];

  const dust = Math.max(0, Math.min(requested, 16384) - planets.length - 1);
  const count = planets.length + 1 + dust;
  const { positions, velocities } = allocate(count);

  positions.set([0, 0, 0, sunMass], 0);

  let cursor = 1;
  for (const planet of planets) {
    const theta = Math.random() * Math.PI * 2;
    const v = circularSpeed(sunMass, planet.r, SEED_SOFTENING);
    const o = cursor * 4;
    positions[o + 0] = Math.cos(theta) * planet.r;
    positions[o + 1] = 0;
    positions[o + 2] = Math.sin(theta) * planet.r;
    positions[o + 3] = planet.m;
    velocities[o + 0] = -Math.sin(theta) * v;
    velocities[o + 1] = 0;
    velocities[o + 2] = Math.cos(theta) * v;
    cursor++;
  }

  for (let i = 0; i < dust; i++) {
    // Two thirds asteroid belt, one third Kuiper belt.
    const inner = i % 3 !== 0;
    const r = inner ? 23 + Math.random() * 5 : 82 + Math.random() * 26;
    const theta = Math.random() * Math.PI * 2;
    const v = circularSpeed(sunMass, r, SEED_SOFTENING) * (1 + gaussian() * 0.01);
    const o = cursor * 4;
    positions[o + 0] = Math.cos(theta) * r;
    positions[o + 1] = gaussian() * (inner ? 0.5 : 2.5);
    positions[o + 2] = Math.sin(theta) * r;
    positions[o + 3] = 1e-4;
    velocities[o + 0] = -Math.sin(theta) * v;
    velocities[o + 1] = 0;
    velocities[o + 2] = Math.cos(theta) * v;
    cursor++;
  }

  return {
    positions,
    velocities,
    count,
    extent: 130,
    referenceSpeed: circularSpeed(sunMass, 16),
    suggestedDt: 0.002,
    label: 'solar system',
  };
}

export function buildPreset(id: PresetId, count: number): BodySet {
  switch (id) {
    case 'galaxy':
      return galaxy(count);
    case 'collision':
      return collision(count);
    case 'sphere':
      return sphere(count);
    case 'solar':
      return solar(count);
  }
}
