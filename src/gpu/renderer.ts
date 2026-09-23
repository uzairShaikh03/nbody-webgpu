import nbodyShader from '../shaders/nbody.wgsl?raw';
import pointsShader from '../shaders/points.wgsl?raw';
import postShader from '../shaders/post.wgsl?raw';
import type { BodyArray, BodySet } from '../presets';

export const MAX_BODIES = 262144;
const WORKGROUP_SIZE = 64;
const ACCUM_FORMAT: GPUTextureFormat = 'rgba16float';

export interface StepParams {
  dt: number;
  g: number;
  softening: number;
  damping: number;
  /** xyz = attractor position, w = mass; mass 0 disables it. */
  attractor: [number, number, number, number];
}

export interface FrameParams {
  viewProjection: Float32Array;
  pointScale: number;
  speedScale: number;
  brightness: number;
  fade: number;
  exposure: number;
}

export class WebGpuUnsupportedError extends Error {}

/**
 * Owns the WebGPU device and every pipeline in the app:
 *
 *   compute (n-body)  ->  posA/posB ping-pong storage buffers
 *   fade              ->  decays the previous HDR accumulation texture
 *   points            ->  additively draws body sprites into the accumulation texture
 *   present           ->  tonemaps the accumulation texture to the swap chain
 */
export class Renderer {
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;

  private readonly posBuffers: [GPUBuffer, GPUBuffer];
  private readonly velBuffer: GPUBuffer;
  private readonly simParamsBuffer: GPUBuffer;
  private readonly renderParamsBuffer: GPUBuffer;
  private readonly postParamsBuffer: GPUBuffer;

  private readonly computePipeline: GPUComputePipeline;
  private readonly pointsPipeline: GPURenderPipeline;
  private readonly fadePipeline: GPURenderPipeline;
  private readonly presentPipeline: GPURenderPipeline;

  private readonly computeBindGroups: [GPUBindGroup, GPUBindGroup];
  private readonly pointsBindGroups: [GPUBindGroup, GPUBindGroup];

  private accumTextures: [GPUTexture, GPUTexture] | null = null;
  private fadeBindGroups: [GPUBindGroup, GPUBindGroup] | null = null;
  private presentBindGroups: [GPUBindGroup, GPUBindGroup] | null = null;

  /** Index of the buffer holding current positions; flips on every compute step. */
  private pingPong = 0;
  /** Index of the accumulation texture holding the latest frame. */
  private accumIndex = 0;
  private bodyCount = 0;

  private readonly simScratch = new ArrayBuffer(48);
  private readonly simFloats = new Float32Array(this.simScratch);
  private readonly simUints = new Uint32Array(this.simScratch);
  private readonly renderScratch = new Float32Array(24);
  private readonly postScratch = new Float32Array(4);

  // GPU-side timing, only if the adapter exposes timestamp queries.
  private readonly timestampQuerySet: GPUQuerySet | null;
  private readonly timestampResolve: GPUBuffer | null;
  private readonly timestampReadback: GPUBuffer | null;
  private timestampPending = false;
  private lastComputeMs: number | null = null;

  readonly adapterInfo: string;
  readonly hasTimestamps: boolean;

  private constructor(
    device: GPUDevice,
    context: GPUCanvasContext,
    canvasFormat: GPUTextureFormat,
    adapterInfo: string,
  ) {
    this.device = device;
    this.context = context;
    this.adapterInfo = adapterInfo;
    this.hasTimestamps = device.features.has('timestamp-query');

    const bodyBytes = MAX_BODIES * 4 * 4;
    const makeBodyBuffer = (label: string) =>
      device.createBuffer({
        label,
        size: bodyBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });

    this.posBuffers = [makeBodyBuffer('positions A'), makeBodyBuffer('positions B')];
    this.velBuffer = makeBodyBuffer('velocities');

    const makeUniform = (label: string, size: number) =>
      device.createBuffer({
        label,
        size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

    this.simParamsBuffer = makeUniform('sim params', 48);
    this.renderParamsBuffer = makeUniform('render params', 96);
    this.postParamsBuffer = makeUniform('post params', 16);

    // --- pipelines ---------------------------------------------------------

    this.computePipeline = device.createComputePipeline({
      label: 'nbody step',
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ label: 'nbody.wgsl', code: nbodyShader }),
        entryPoint: 'main',
      },
    });

    const pointsModule = device.createShaderModule({ label: 'points.wgsl', code: pointsShader });
    this.pointsPipeline = device.createRenderPipeline({
      label: 'body sprites',
      layout: 'auto',
      vertex: { module: pointsModule, entryPoint: 'vs' },
      fragment: {
        module: pointsModule,
        entryPoint: 'fs',
        targets: [
          {
            format: ACCUM_FORMAT,
            // Pure additive: overlapping sprites sum into HDR density, so the
            // draw needs no depth buffer and no back-to-front sorting.
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list' },
    });

    const postModule = device.createShaderModule({ label: 'post.wgsl', code: postShader });
    this.fadePipeline = device.createRenderPipeline({
      label: 'trail fade',
      layout: 'auto',
      vertex: { module: postModule, entryPoint: 'vs' },
      fragment: { module: postModule, entryPoint: 'fsFade', targets: [{ format: ACCUM_FORMAT }] },
      primitive: { topology: 'triangle-list' },
    });

    this.presentPipeline = device.createRenderPipeline({
      label: 'tonemap present',
      layout: 'auto',
      vertex: { module: postModule, entryPoint: 'vs' },
      fragment: {
        module: postModule,
        entryPoint: 'fsPresent',
        targets: [{ format: canvasFormat }],
      },
      primitive: { topology: 'triangle-list' },
    });

    // --- bind groups (one per ping-pong phase) -----------------------------

    const computeLayout = this.computePipeline.getBindGroupLayout(0);
    this.computeBindGroups = [0, 1].map((i) =>
      device.createBindGroup({
        label: `compute bind group ${i}`,
        layout: computeLayout,
        entries: [
          { binding: 0, resource: { buffer: this.posBuffers[i] } },
          { binding: 1, resource: { buffer: this.posBuffers[1 - i] } },
          { binding: 2, resource: { buffer: this.velBuffer } },
          { binding: 3, resource: { buffer: this.simParamsBuffer } },
        ],
      }),
    ) as [GPUBindGroup, GPUBindGroup];

    const pointsLayout = this.pointsPipeline.getBindGroupLayout(0);
    this.pointsBindGroups = [0, 1].map((i) =>
      device.createBindGroup({
        label: `points bind group ${i}`,
        layout: pointsLayout,
        entries: [
          { binding: 0, resource: { buffer: this.posBuffers[i] } },
          { binding: 1, resource: { buffer: this.velBuffer } },
          { binding: 2, resource: { buffer: this.renderParamsBuffer } },
        ],
      }),
    ) as [GPUBindGroup, GPUBindGroup];

    if (this.hasTimestamps) {
      this.timestampQuerySet = device.createQuerySet({ type: 'timestamp', count: 2 });
      this.timestampResolve = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.timestampReadback = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    } else {
      this.timestampQuerySet = null;
      this.timestampResolve = null;
      this.timestampReadback = null;
    }
  }

  static async create(canvas: HTMLCanvasElement): Promise<Renderer> {
    if (!navigator.gpu) {
      throw new WebGpuUnsupportedError('navigator.gpu is undefined — this browser has no WebGPU.');
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      throw new WebGpuUnsupportedError('requestAdapter() returned null — no compatible GPU adapter.');
    }

    // timestamp-query is optional; ask for it only when the adapter has it.
    const requiredFeatures: GPUFeatureName[] = adapter.features.has('timestamp-query')
      ? ['timestamp-query']
      : [];

    const device = await adapter.requestDevice({ requiredFeatures });
    device.lost.then((info) => {
      console.error('WebGPU device lost:', info.reason, info.message);
    });
    device.addEventListener('uncapturederror', (event) => {
      console.error('WebGPU error:', (event as GPUUncapturedErrorEvent).error.message);
    });

    const context = canvas.getContext('webgpu');
    if (!context) {
      throw new WebGpuUnsupportedError('canvas.getContext("webgpu") returned null.');
    }

    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format: canvasFormat, alphaMode: 'opaque' });

    const info = adapter.info ?? (await (adapter as unknown as { requestAdapterInfo?: () => Promise<GPUAdapterInfo> }).requestAdapterInfo?.());
    const describe = [info?.vendor, info?.architecture].filter(Boolean).join(' ') || 'gpu';

    return new Renderer(device, context, canvasFormat, describe);
  }

  get count(): number {
    return this.bodyCount;
  }

  /** Uploads a fresh scenario and resets the ping-pong state. */
  setBodies(bodies: BodySet): void {
    this.bodyCount = bodies.count;
    this.pingPong = 0;
    this.device.queue.writeBuffer(this.posBuffers[0], 0, bodies.positions, 0, bodies.count * 4);
    this.device.queue.writeBuffer(this.velBuffer, 0, bodies.velocities, 0, bodies.count * 4);
    this.clearTrails();
  }

  /** Overwrites GPU state from CPU-side arrays (used by the CPU solver modes). */
  uploadState(positions: BodyArray, velocities: BodyArray, count: number): void {
    this.bodyCount = count;
    this.device.queue.writeBuffer(this.posBuffers[this.pingPong], 0, positions, 0, count * 4);
    this.device.queue.writeBuffer(this.velBuffer, 0, velocities, 0, count * 4);
  }

  /** Encodes and submits `substeps` n-body integrations. */
  step(params: StepParams, substeps = 1): void {
    if (this.bodyCount === 0) return;

    this.simFloats[0] = params.attractor[0];
    this.simFloats[1] = params.attractor[1];
    this.simFloats[2] = params.attractor[2];
    this.simFloats[3] = params.attractor[3];
    this.simFloats[4] = params.dt;
    this.simFloats[5] = params.g;
    this.simFloats[6] = params.softening * params.softening;
    this.simFloats[7] = params.damping;
    this.simUints[8] = this.bodyCount;
    this.device.queue.writeBuffer(this.simParamsBuffer, 0, this.simScratch);

    const encoder = this.device.createCommandEncoder({ label: 'nbody steps' });
    const workgroups = Math.ceil(this.bodyCount / WORKGROUP_SIZE);

    for (let i = 0; i < substeps; i++) {
      const timestamps =
        i === 0 && this.timestampQuerySet && !this.timestampPending
          ? {
              querySet: this.timestampQuerySet,
              beginningOfPassWriteIndex: 0,
              endOfPassWriteIndex: 1,
            }
          : undefined;

      const pass = encoder.beginComputePass({ timestampWrites: timestamps });
      pass.setPipeline(this.computePipeline);
      pass.setBindGroup(0, this.computeBindGroups[this.pingPong]);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
      this.pingPong = 1 - this.pingPong;
    }

    if (this.timestampQuerySet && this.timestampResolve && this.timestampReadback && !this.timestampPending) {
      encoder.resolveQuerySet(this.timestampQuerySet, 0, 2, this.timestampResolve, 0);
      encoder.copyBufferToBuffer(this.timestampResolve, 0, this.timestampReadback, 0, 16);
      this.timestampPending = true;
      this.device.queue.submit([encoder.finish()]);
      void this.readTimestamps();
      return;
    }

    this.device.queue.submit([encoder.finish()]);
  }

  private async readTimestamps(): Promise<void> {
    const buffer = this.timestampReadback;
    if (!buffer) return;
    try {
      await buffer.mapAsync(GPUMapMode.READ);
      const times = new BigUint64Array(buffer.getMappedRange().slice(0));
      buffer.unmap();
      const deltaNs = Number(times[1] - times[0]);
      if (deltaNs > 0) {
        this.lastComputeMs = deltaNs / 1e6;
      }
    } catch {
      // Device busy or lost; skip this sample.
    } finally {
      this.timestampPending = false;
    }
  }

  /** Most recent GPU-measured compute pass duration in ms, if available. */
  get computeMs(): number | null {
    return this.lastComputeMs;
  }

  /** Blocks until all submitted GPU work has completed (used for benchmarking). */
  async waitForGpu(): Promise<void> {
    await this.device.queue.onSubmittedWorkDone();
  }

  resize(width: number, height: number): void {
    const canvas = this.context.canvas as HTMLCanvasElement;
    if (canvas.width === width && canvas.height === height && this.accumTextures) return;

    canvas.width = width;
    canvas.height = height;

    for (const texture of this.accumTextures ?? []) {
      texture.destroy();
    }

    const makeAccum = (label: string) =>
      this.device.createTexture({
        label,
        size: { width, height },
        format: ACCUM_FORMAT,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT |
          GPUTextureUsage.TEXTURE_BINDING |
          // COPY_SRC so the self-test can read pixels back through the GPU,
          // independent of browser screenshot/compositing paths.
          GPUTextureUsage.COPY_SRC,
      });

    this.accumTextures = [makeAccum('accum A'), makeAccum('accum B')];

    const fadeLayout = this.fadePipeline.getBindGroupLayout(0);
    const presentLayout = this.presentPipeline.getBindGroupLayout(0);

    this.fadeBindGroups = [0, 1].map((i) =>
      this.device.createBindGroup({
        layout: fadeLayout,
        entries: [
          // Reads the *other* texture: the previous frame's accumulation.
          { binding: 0, resource: this.accumTextures![1 - i].createView() },
          { binding: 1, resource: { buffer: this.postParamsBuffer } },
        ],
      }),
    ) as [GPUBindGroup, GPUBindGroup];

    this.presentBindGroups = [0, 1].map((i) =>
      this.device.createBindGroup({
        layout: presentLayout,
        entries: [
          { binding: 0, resource: this.accumTextures![i].createView() },
          { binding: 1, resource: { buffer: this.postParamsBuffer } },
        ],
      }),
    ) as [GPUBindGroup, GPUBindGroup];
  }

  /** Wipes the trail history, e.g. after re-seeding a scenario. */
  clearTrails(): void {
    if (!this.accumTextures) return;
    const encoder = this.device.createCommandEncoder({ label: 'clear trails' });
    for (const texture of this.accumTextures) {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: texture.createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });
      pass.end();
    }
    this.device.queue.submit([encoder.finish()]);
  }

  render(frame: FrameParams): void {
    if (!this.accumTextures || !this.fadeBindGroups || !this.presentBindGroups) return;

    const canvas = this.context.canvas as HTMLCanvasElement;
    const target = 1 - this.accumIndex;

    this.renderScratch.set(frame.viewProjection, 0);
    this.renderScratch[16] = canvas.width;
    this.renderScratch[17] = canvas.height;
    this.renderScratch[18] = frame.pointScale;
    this.renderScratch[19] = 0.7;
    this.renderScratch[20] = 64;
    this.renderScratch[21] = frame.speedScale;
    this.renderScratch[22] = frame.brightness;
    this.device.queue.writeBuffer(this.renderParamsBuffer, 0, this.renderScratch);

    this.postScratch[0] = frame.fade;
    this.postScratch[1] = frame.exposure;
    this.postScratch[2] = 0.35;
    this.device.queue.writeBuffer(this.postParamsBuffer, 0, this.postScratch);

    const encoder = this.device.createCommandEncoder({ label: 'frame' });
    const targetView = this.accumTextures[target].createView();

    // 1. Copy the previous accumulation scaled by `fade` into the new target.
    const fadePass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: targetView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    fadePass.setPipeline(this.fadePipeline);
    fadePass.setBindGroup(0, this.fadeBindGroups[target]);
    fadePass.draw(3);
    fadePass.end();

    // 2. Additively draw this frame's bodies on top of the faded history.
    const pointsPass = encoder.beginRenderPass({
      colorAttachments: [{ view: targetView, loadOp: 'load', storeOp: 'store' }],
    });
    pointsPass.setPipeline(this.pointsPipeline);
    pointsPass.setBindGroup(0, this.pointsBindGroups[this.pingPong]);
    pointsPass.draw(6, this.bodyCount);
    pointsPass.end();

    // 3. Tonemap to the swap chain.
    const presentPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    presentPass.setPipeline(this.presentPipeline);
    presentPass.setBindGroup(0, this.presentBindGroups[target]);
    presentPass.draw(3);
    presentPass.end();

    this.device.queue.submit([encoder.finish()]);
    this.accumIndex = target;
  }

  // -------------------------------------------------------------------------
  // Self-test helpers.
  //
  // Headless screenshots cannot capture a WebGPU surface on every platform, so
  // correctness is verified by pulling data back through the GPU instead of
  // trusting a captured image.
  // -------------------------------------------------------------------------

  /** Reads the current body positions (xyzw per body) back to the CPU. */
  async readPositions(count: number): Promise<Float32Array> {
    const bytes = count * 16;
    const staging = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: 'read positions' });
    encoder.copyBufferToBuffer(this.posBuffers[this.pingPong], 0, staging, 0, bytes);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return copy;
  }

  /** Samples a centred region of the HDR accumulation buffer and summarizes it. */
  async probePixels(): Promise<{ maxLuma: number; avgLuma: number; litFraction: number }> {
    if (!this.accumTextures) throw new Error('no accumulation texture yet');
    const canvas = this.context.canvas as HTMLCanvasElement;
    const width = Math.min(512, canvas.width);
    const height = Math.min(288, canvas.height);
    // rgba16float is 8 bytes per pixel and copies require a 256-byte row stride.
    const bytesPerRow = Math.ceil((width * 8) / 256) * 256;

    const staging = this.device.createBuffer({
      size: bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = this.device.createCommandEncoder({ label: 'probe pixels' });
    encoder.copyTextureToBuffer(
      {
        texture: this.accumTextures[this.accumIndex],
        origin: {
          x: Math.floor((canvas.width - width) / 2),
          y: Math.floor((canvas.height - height) / 2),
        },
      },
      { buffer: staging, bytesPerRow, rowsPerImage: height },
      { width, height },
    );
    this.device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const halves = new Uint16Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();

    let maxLuma = 0;
    let total = 0;
    let lit = 0;
    const rowStride = bytesPerRow / 2;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = y * rowStride + x * 4;
        const luma =
          (halfToFloat(halves[i]) + halfToFloat(halves[i + 1]) + halfToFloat(halves[i + 2])) / 3;
        total += luma;
        if (luma > maxLuma) maxLuma = luma;
        if (luma > 0.004) lit++;
      }
    }

    return {
      maxLuma,
      avgLuma: total / (width * height),
      litFraction: lit / (width * height),
    };
  }
}

/** Decodes an IEEE 754 binary16 value stored in a uint16. */
function halfToFloat(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x03ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 31) return mantissa ? NaN : sign * Infinity;
  return sign * (mantissa + 1024) * 2 ** (exponent - 25);
}
