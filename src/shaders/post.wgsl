// ---------------------------------------------------------------------------
// Post passes over the HDR accumulation buffer.
//
// Motion trails come from ping-ponging two rgba16float targets: the fade pass
// copies the previous frame scaled by `fade`, the point pass adds this frame's
// sprites on top, and the present pass tonemaps the result to the swap chain.
// A geometric decay per frame is what produces the exponential comet tails.
// ---------------------------------------------------------------------------

struct PostParams {
  fade: f32,
  exposure: f32,
  vignette: f32,
  _pad0: f32,
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: PostParams;

struct VSOut {
  @builtin(position) clip: vec4f,
};

// Oversized triangle covering the viewport; avoids binding any vertex buffers.
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var verts = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0),
  );
  var out: VSOut;
  out.clip = vec4f(verts[vi], 0.0, 1.0);
  return out;
}

@fragment
fn fsFade(in: VSOut) -> @location(0) vec4f {
  let texel = textureLoad(src, vec2i(in.clip.xy), 0);
  return vec4f(texel.rgb * params.fade, texel.a * params.fade);
}

@fragment
fn fsPresent(in: VSOut) -> @location(0) vec4f {
  let dims = vec2f(textureDimensions(src));
  let uv = in.clip.xy / dims;

  var hdr = textureLoad(src, vec2i(in.clip.xy), 0).rgb * params.exposure;

  // Exponential tonemap: keeps dense cores from clipping to flat white.
  var color = vec3f(1.0) - exp(-hdr);

  // Faint cool background gradient so empty space is not pure black.
  let centered = uv - vec2f(0.5);
  let radial = length(centered);
  color += vec3f(0.012, 0.018, 0.038) * (1.0 - radial);

  // Vignette.
  color *= 1.0 - params.vignette * smoothstep(0.35, 0.95, radial);

  // Linear -> sRGB-ish for an 8-bit non-sRGB swap chain.
  color = pow(max(color, vec3f(0.0)), vec3f(1.0 / 2.2));
  return vec4f(color, 1.0);
}
