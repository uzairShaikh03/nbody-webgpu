// ---------------------------------------------------------------------------
// Point-sprite renderer.
//
// WebGPU has no gl_PointSize, so each body is drawn as a two-triangle billboard
// (6 vertices, 1 instance per body) expanded in clip space. The vertex shader
// reads positions and velocities straight out of the same storage buffers the
// compute pass writes, so simulation data never round-trips through the CPU.
//
// Sprites are additively blended with a soft radial falloff: overlapping bodies
// accumulate into bloom-like density instead of z-fighting, which also means the
// draw is order independent and needs no depth buffer or sorting.
// ---------------------------------------------------------------------------

struct RenderParams {
  viewProj: mat4x4f,
  viewport: vec2f,
  // World radius pre-multiplied by focal length in pixels, so dividing by clip.w
  // yields a perspective-correct on-screen radius.
  pointScale: f32,
  minPx: f32,
  maxPx: f32,
  // 1 / reference speed: normalizes velocity into the colour ramp.
  speedScale: f32,
  brightness: f32,
  // Pads the struct to 96 bytes (16-byte alignment). Getting this wrong makes
  // the uniform buffer too small for the layout and the bind group fails
  // validation, which silently discards every pass in the frame.
  _pad0: f32,
};

@group(0) @binding(0) var<storage, read> pos: array<vec4f>;
@group(0) @binding(1) var<storage, read> vel: array<vec4f>;
@group(0) @binding(2) var<uniform> params: RenderParams;

struct VSOut {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
  @location(1) color: vec3f,
};

fn speedColor(t: f32) -> vec3f {
  let cold = vec3f(0.22, 0.42, 1.0);
  let mid = vec3f(0.72, 0.86, 1.0);
  let hot = vec3f(1.0, 0.72, 0.32);
  if (t < 0.5) {
    return mix(cold, mid, t * 2.0);
  }
  return mix(mid, hot, (t - 0.5) * 2.0);
}

@vertex
fn vs(
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) ii: u32,
) -> VSOut {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0),
    vec2f(1.0, 1.0),
  );
  let corner = corners[vi];

  let body = pos[ii];
  var clip = params.viewProj * vec4f(body.xyz, 1.0);

  // Heavier bodies draw slightly larger (cube root keeps it subtle).
  let massScale = pow(max(body.w, 1e-4), 0.3333);
  let radiusPx = clamp(params.pointScale * massScale / max(clip.w, 1e-4), params.minPx, params.maxPx);

  // Pixels -> clip space: NDC spans 2 units across `viewport` pixels, and clip = ndc * w.
  clip = vec4f(
    clip.xy + corner * (radiusPx * 2.0 / params.viewport) * clip.w,
    clip.zw,
  );

  let speed = length(vel[ii].xyz);
  let t = clamp(speed * params.speedScale, 0.0, 1.0);

  var out: VSOut;
  out.clip = clip;
  out.uv = corner;
  out.color = speedColor(t);
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let r = length(in.uv);
  if (r > 1.0) {
    discard;
  }
  // Soft core with a long tail; squaring keeps centres bright while edges fade.
  let falloff = pow(1.0 - r, 2.5);
  return vec4f(in.color * falloff * params.brightness, falloff);
}
