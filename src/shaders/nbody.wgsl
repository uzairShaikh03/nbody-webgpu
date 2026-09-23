// ---------------------------------------------------------------------------
// Gravitational N-body integrator.
//
// Every body feels every other body, so a step is O(n^2) force evaluations.
// The win here is memory traffic, not arithmetic: a naive kernel re-reads all n
// positions from global storage for each of the n threads (n^2 global reads).
// Instead each workgroup cooperatively stages a TILE-sized block of bodies into
// on-chip `var<workgroup>` memory, and all TILE threads in the group reuse that
// block. Global reads drop to n^2 / TILE, which is where the speedup comes from
// on memory-bound hardware.
//
// Positions are double-buffered (ping-pong): a thread reads every body's current
// position while writing only its own new one, so reading and writing the same
// buffer would be a data race. Velocities are touched only by their owning
// thread, so a single velocity buffer is safe.
// ---------------------------------------------------------------------------

struct SimParams {
  // xyz = world position of the user-dropped attractor, w = its mass (0 = off).
  attractor: vec4f,
  dt: f32,
  g: f32,
  // Plummer softening, pre-squared. Keeps close encounters from producing
  // near-infinite accelerations that would blow the integrator up.
  softening2: f32,
  damping: f32,
  count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

// xyz = position, w = mass
@group(0) @binding(0) var<storage, read> posIn: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> posOut: array<vec4f>;
// xyz = velocity, w = |acceleration| from the previous step (used for shading)
@group(0) @binding(2) var<storage, read_write> vel: array<vec4f>;
@group(0) @binding(3) var<uniform> params: SimParams;

const TILE: u32 = 64u;

var<workgroup> tile: array<vec4f, TILE>;

@compute @workgroup_size(TILE)
fn main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let i = gid.x;

  // Threads past the end of the body list still have to reach every
  // workgroupBarrier() below (barriers must be hit in uniform control flow), so
  // instead of returning early they do harmless duplicate work and skip the
  // final write.
  let selfIndex = min(i, max(params.count, 1u) - 1u);
  let selfPos = posIn[selfIndex].xyz;

  var acc = vec3f(0.0);
  let tileCount = (params.count + TILE - 1u) / TILE;

  for (var t = 0u; t < tileCount; t = t + 1u) {
    let src = t * TILE + lid.x;
    // Out-of-range slots get mass 0, contributing no force.
    tile[lid.x] = select(vec4f(0.0), posIn[src], src < params.count);
    workgroupBarrier();

    for (var k = 0u; k < TILE; k = k + 1u) {
      let other = tile[k];
      let d = other.xyz - selfPos;
      // a = G * m * d / (|d|^2 + eps^2)^(3/2)
      let dist2 = dot(d, d) + params.softening2;
      let invDist = inverseSqrt(dist2);
      let invDist3 = invDist * invDist * invDist;
      acc += d * (other.w * invDist3);
    }
    workgroupBarrier();
  }

  if (params.attractor.w > 0.0) {
    let d = params.attractor.xyz - selfPos;
    let dist2 = dot(d, d) + params.softening2;
    let invDist = inverseSqrt(dist2);
    acc += d * (params.attractor.w * invDist * invDist * invDist);
  }

  acc *= params.g;

  if (i < params.count) {
    // Semi-implicit (symplectic) Euler: velocity first, then position with the
    // already-updated velocity. Cheap, and far more stable than explicit Euler
    // for orbital motion.
    let v = (vel[i].xyz + acc * params.dt) * params.damping;
    vel[i] = vec4f(v, length(acc));
    posOut[i] = vec4f(selfPos + v * params.dt, posIn[i].w);
  }
}
