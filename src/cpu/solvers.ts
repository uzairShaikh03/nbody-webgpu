export interface CpuStepParams {
  dt: number;
  g: number;
  softening: number;
  damping: number;
  theta: number;
  attractor: [number, number, number, number];
}

export interface CpuSolver {
  readonly label: string;
  /** Advances state in place. Arrays are xyzw-interleaved, 4 floats per body. */
  step(positions: Float32Array, velocities: Float32Array, count: number, params: CpuStepParams): void;
  /** Force evaluations performed in the last step, i.e. the real algorithmic cost. */
  readonly lastInteractions: number;
  /** Octree nodes built in the last step (Barnes-Hut only). */
  readonly lastNodes: number;
}

function applyAttractor(
  positions: Float32Array,
  accelerations: Float32Array,
  count: number,
  attractor: [number, number, number, number],
  softening2: number,
): void {
  if (attractor[3] <= 0) return;
  for (let i = 0; i < count; i++) {
    const dx = attractor[0] - positions[i * 4];
    const dy = attractor[1] - positions[i * 4 + 1];
    const dz = attractor[2] - positions[i * 4 + 2];
    const dist2 = dx * dx + dy * dy + dz * dz + softening2;
    const invDist3 = 1 / (dist2 * Math.sqrt(dist2));
    const f = attractor[3] * invDist3;
    accelerations[i * 3] += dx * f;
    accelerations[i * 3 + 1] += dy * f;
    accelerations[i * 3 + 2] += dz * f;
  }
}

function integrate(
  positions: Float32Array,
  velocities: Float32Array,
  accelerations: Float32Array,
  count: number,
  params: CpuStepParams,
): void {
  const { dt, g, damping } = params;
  for (let i = 0; i < count; i++) {
    const p = i * 4;
    const a = i * 3;
    const vx = (velocities[p] + accelerations[a] * g * dt) * damping;
    const vy = (velocities[p + 1] + accelerations[a + 1] * g * dt) * damping;
    const vz = (velocities[p + 2] + accelerations[a + 2] * g * dt) * damping;
    velocities[p] = vx;
    velocities[p + 1] = vy;
    velocities[p + 2] = vz;
    velocities[p + 3] = Math.hypot(accelerations[a], accelerations[a + 1], accelerations[a + 2]) * g;
    positions[p] += vx * dt;
    positions[p + 1] += vy * dt;
    positions[p + 2] += vz * dt;
  }
}

/**
 * Direct summation, the reference implementation.
 *
 * Uses Newton's third law: each pair is evaluated once and the force applied to
 * both bodies, halving the work versus a full double loop. This is deliberately
 * the *fast* CPU version, so the GPU comparison is honest.
 */
export class NaiveSolver implements CpuSolver {
  readonly label = 'CPU naive O(n²)';
  lastInteractions = 0;
  lastNodes = 0;

  private acc = new Float32Array(0);

  step(positions: Float32Array, velocities: Float32Array, count: number, params: CpuStepParams): void {
    if (this.acc.length < count * 3) this.acc = new Float32Array(count * 3);
    const acc = this.acc;
    acc.fill(0, 0, count * 3);

    const softening2 = params.softening * params.softening;

    for (let i = 0; i < count; i++) {
      const pi = i * 4;
      const xi = positions[pi];
      const yi = positions[pi + 1];
      const zi = positions[pi + 2];
      const mi = positions[pi + 3];
      const ai = i * 3;
      let axi = 0;
      let ayi = 0;
      let azi = 0;

      for (let j = i + 1; j < count; j++) {
        const pj = j * 4;
        const dx = positions[pj] - xi;
        const dy = positions[pj + 1] - yi;
        const dz = positions[pj + 2] - zi;
        const dist2 = dx * dx + dy * dy + dz * dz + softening2;
        const invDist3 = 1 / (dist2 * Math.sqrt(dist2));

        const fj = positions[pj + 3] * invDist3;
        axi += dx * fj;
        ayi += dy * fj;
        azi += dz * fj;

        const fi = mi * invDist3;
        const aj = j * 3;
        acc[aj] -= dx * fi;
        acc[aj + 1] -= dy * fi;
        acc[aj + 2] -= dz * fi;
      }

      acc[ai] += axi;
      acc[ai + 1] += ayi;
      acc[ai + 2] += azi;
    }

    this.lastInteractions = (count * (count - 1)) / 2;
    applyAttractor(positions, acc, count, params.attractor, softening2);
    integrate(positions, velocities, acc, count, params);
  }
}

const LEAF_CAPACITY = 8;
const MAX_DEPTH = 24;
const CHILD_STRIDE = 8;

/**
 * Barnes-Hut octree solver, O(n log n).
 *
 * Bodies are bucketed into an octree; when computing the force on a body, any
 * node whose angular size is below θ is collapsed to its centre of mass instead
 * of being opened. Tuning θ trades accuracy for speed: θ = 0 degenerates to
 * direct summation, larger θ approximates more aggressively.
 *
 * The tree is stored in flat typed arrays and rebuilt every step, with bodies
 * threaded through leaves as an intrusive linked list (`nextBody`), so a step
 * allocates nothing once the arrays are warm.
 */
export class BarnesHutSolver implements CpuSolver {
  readonly label = 'CPU Barnes-Hut O(n log n)';
  lastInteractions = 0;
  lastNodes = 0;

  private acc = new Float32Array(0);
  private nextBody = new Int32Array(0);

  // Flat octree storage.
  private capacity = 0;
  private child = new Int32Array(0);
  private centerX = new Float32Array(0);
  private centerY = new Float32Array(0);
  private centerZ = new Float32Array(0);
  private half = new Float32Array(0);
  private mass = new Float32Array(0);
  private comX = new Float32Array(0);
  private comY = new Float32Array(0);
  private comZ = new Float32Array(0);
  private firstBody = new Int32Array(0);
  private bodyCount = new Int32Array(0);
  private isLeaf = new Uint8Array(0);
  private depthOf = new Uint8Array(0);
  private nodeCount = 0;
  private stack = new Int32Array(1024);

  private ensureCapacity(nodes: number): void {
    if (nodes <= this.capacity) return;
    const capacity = Math.max(1024, nodes * 2);
    const growI32 = (src: Int32Array, stride = 1) => {
      const next = new Int32Array(capacity * stride);
      next.set(src);
      return next;
    };
    const growF32 = (src: Float32Array) => {
      const next = new Float32Array(capacity);
      next.set(src);
      return next;
    };
    this.child = growI32(this.child, CHILD_STRIDE);
    this.centerX = growF32(this.centerX);
    this.centerY = growF32(this.centerY);
    this.centerZ = growF32(this.centerZ);
    this.half = growF32(this.half);
    this.mass = growF32(this.mass);
    this.comX = growF32(this.comX);
    this.comY = growF32(this.comY);
    this.comZ = growF32(this.comZ);
    this.firstBody = growI32(this.firstBody);
    this.bodyCount = growI32(this.bodyCount);
    const leaf = new Uint8Array(capacity);
    leaf.set(this.isLeaf);
    this.isLeaf = leaf;
    const depth = new Uint8Array(capacity);
    depth.set(this.depthOf);
    this.depthOf = depth;
    this.capacity = capacity;
  }

  private newNode(cx: number, cy: number, cz: number, halfSize: number, depth: number): number {
    this.ensureCapacity(this.nodeCount + 1);
    const node = this.nodeCount++;
    this.centerX[node] = cx;
    this.centerY[node] = cy;
    this.centerZ[node] = cz;
    this.half[node] = halfSize;
    this.mass[node] = 0;
    this.comX[node] = 0;
    this.comY[node] = 0;
    this.comZ[node] = 0;
    this.firstBody[node] = -1;
    this.bodyCount[node] = 0;
    this.isLeaf[node] = 1;
    this.depthOf[node] = depth;
    this.child.fill(-1, node * CHILD_STRIDE, node * CHILD_STRIDE + CHILD_STRIDE);
    return node;
  }

  private octantOf(node: number, x: number, y: number, z: number): number {
    return (
      (x > this.centerX[node] ? 1 : 0) |
      (y > this.centerY[node] ? 2 : 0) |
      (z > this.centerZ[node] ? 4 : 0)
    );
  }

  private childNode(node: number, octant: number): number {
    const slot = node * CHILD_STRIDE + octant;
    let existing = this.child[slot];
    if (existing < 0) {
      const quarter = this.half[node] * 0.5;
      existing = this.newNode(
        this.centerX[node] + (octant & 1 ? quarter : -quarter),
        this.centerY[node] + (octant & 2 ? quarter : -quarter),
        this.centerZ[node] + (octant & 4 ? quarter : -quarter),
        quarter,
        this.depthOf[node] + 1,
      );
      // newNode may have reallocated `child`; index again.
      this.child[node * CHILD_STRIDE + octant] = existing;
    }
    return existing;
  }

  private insert(root: number, body: number, positions: Float32Array): void {
    const x = positions[body * 4];
    const y = positions[body * 4 + 1];
    const z = positions[body * 4 + 2];

    let node = root;
    while (this.isLeaf[node] === 0) {
      node = this.childNode(node, this.octantOf(node, x, y, z));
    }

    this.nextBody[body] = this.firstBody[node];
    this.firstBody[node] = body;
    this.bodyCount[node]++;

    if (this.bodyCount[node] > LEAF_CAPACITY && this.depthOf[node] < MAX_DEPTH) {
      this.subdivide(node, positions);
    }
  }

  /** Converts a full leaf into an internal node and pushes its bodies down. */
  private subdivide(node: number, positions: Float32Array): void {
    let body = this.firstBody[node];
    this.firstBody[node] = -1;
    this.bodyCount[node] = 0;
    this.isLeaf[node] = 0;

    while (body >= 0) {
      const next = this.nextBody[body];
      const octant = this.octantOf(
        node,
        positions[body * 4],
        positions[body * 4 + 1],
        positions[body * 4 + 2],
      );
      const target = this.childNode(node, octant);
      this.nextBody[body] = this.firstBody[target];
      this.firstBody[target] = body;
      this.bodyCount[target]++;
      body = next;
    }

    for (let octant = 0; octant < CHILD_STRIDE; octant++) {
      const target = this.child[node * CHILD_STRIDE + octant];
      if (
        target >= 0 &&
        this.isLeaf[target] === 1 &&
        this.bodyCount[target] > LEAF_CAPACITY &&
        this.depthOf[target] < MAX_DEPTH
      ) {
        this.subdivide(target, positions);
      }
    }
  }

  private build(positions: Float32Array, count: number): number {
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    for (let i = 0; i < count; i++) {
      const p = i * 4;
      if (positions[p] < minX) minX = positions[p];
      if (positions[p] > maxX) maxX = positions[p];
      if (positions[p + 1] < minY) minY = positions[p + 1];
      if (positions[p + 1] > maxY) maxY = positions[p + 1];
      if (positions[p + 2] < minZ) minZ = positions[p + 2];
      if (positions[p + 2] > maxZ) maxZ = positions[p + 2];
    }

    const halfSize =
      Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-3) * 0.5 + 1e-3;

    this.nodeCount = 0;
    // Rough upper bound so insertion rarely reallocates mid-build.
    this.ensureCapacity(Math.max(1024, count));
    const root = this.newNode(
      (minX + maxX) * 0.5,
      (minY + maxY) * 0.5,
      (minZ + maxZ) * 0.5,
      halfSize,
      0,
    );

    for (let i = 0; i < count; i++) {
      this.insert(root, i, positions);
    }
    return root;
  }

  /**
   * Bottom-up centre-of-mass pass. Children are always allocated after their
   * parent, so iterating node indices in reverse visits every child first.
   */
  private computeCentersOfMass(positions: Float32Array): void {
    for (let node = this.nodeCount - 1; node >= 0; node--) {
      let m = 0;
      let cx = 0;
      let cy = 0;
      let cz = 0;

      if (this.isLeaf[node] === 1) {
        for (let body = this.firstBody[node]; body >= 0; body = this.nextBody[body]) {
          const p = body * 4;
          const bm = positions[p + 3];
          m += bm;
          cx += positions[p] * bm;
          cy += positions[p + 1] * bm;
          cz += positions[p + 2] * bm;
        }
      } else {
        for (let octant = 0; octant < CHILD_STRIDE; octant++) {
          const kid = this.child[node * CHILD_STRIDE + octant];
          if (kid < 0) continue;
          const km = this.mass[kid];
          m += km;
          cx += this.comX[kid] * km;
          cy += this.comY[kid] * km;
          cz += this.comZ[kid] * km;
        }
      }

      this.mass[node] = m;
      if (m > 0) {
        this.comX[node] = cx / m;
        this.comY[node] = cy / m;
        this.comZ[node] = cz / m;
      }
    }
  }

  step(positions: Float32Array, velocities: Float32Array, count: number, params: CpuStepParams): void {
    if (this.acc.length < count * 3) this.acc = new Float32Array(count * 3);
    if (this.nextBody.length < count) this.nextBody = new Int32Array(count);
    const acc = this.acc;
    acc.fill(0, 0, count * 3);

    const root = this.build(positions, count);
    this.computeCentersOfMass(positions);

    if (this.stack.length < this.nodeCount + 64) {
      this.stack = new Int32Array((this.nodeCount + 64) * 2);
    }

    const softening2 = params.softening * params.softening;
    const theta2 = params.theta * params.theta;
    let interactions = 0;

    for (let i = 0; i < count; i++) {
      const p = i * 4;
      const xi = positions[p];
      const yi = positions[p + 1];
      const zi = positions[p + 2];
      let ax = 0;
      let ay = 0;
      let az = 0;

      let top = 0;
      this.stack[top++] = root;

      while (top > 0) {
        const node = this.stack[--top];
        const nodeMass = this.mass[node];
        if (nodeMass <= 0) continue;

        const dx = this.comX[node] - xi;
        const dy = this.comY[node] - yi;
        const dz = this.comZ[node] - zi;
        const dist2 = dx * dx + dy * dy + dz * dz;
        const width = this.half[node] * 2;

        // Opening criterion: width / distance < θ  (squared to avoid a sqrt).
        if (this.isLeaf[node] === 0 && width * width >= theta2 * dist2) {
          for (let octant = 0; octant < CHILD_STRIDE; octant++) {
            const kid = this.child[node * CHILD_STRIDE + octant];
            if (kid >= 0) this.stack[top++] = kid;
          }
          continue;
        }

        if (this.isLeaf[node] === 1 && width * width >= theta2 * dist2) {
          // Close leaf: sum its bodies exactly.
          for (let body = this.firstBody[node]; body >= 0; body = this.nextBody[body]) {
            if (body === i) continue;
            const q = body * 4;
            const bx = positions[q] - xi;
            const by = positions[q + 1] - yi;
            const bz = positions[q + 2] - zi;
            const d2 = bx * bx + by * by + bz * bz + softening2;
            const invDist3 = 1 / (d2 * Math.sqrt(d2));
            const f = positions[q + 3] * invDist3;
            ax += bx * f;
            ay += by * f;
            az += bz * f;
            interactions++;
          }
          continue;
        }

        // Far enough: the whole node acts as a single mass at its centre of mass.
        const d2 = dist2 + softening2;
        const invDist3 = 1 / (d2 * Math.sqrt(d2));
        const f = nodeMass * invDist3;
        ax += dx * f;
        ay += dy * f;
        az += dz * f;
        interactions++;
      }

      const a = i * 3;
      acc[a] = ax;
      acc[a + 1] = ay;
      acc[a + 2] = az;
    }

    this.lastInteractions = interactions;
    this.lastNodes = this.nodeCount;
    applyAttractor(positions, acc, count, params.attractor, softening2);
    integrate(positions, velocities, acc, count, params);
  }
}
