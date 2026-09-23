import './style.css';
import { OrbitCamera } from './camera';
import { BarnesHutSolver, NaiveSolver, type CpuSolver, type CpuStepParams } from './cpu/solvers';
import { MAX_BODIES, Renderer, WebGpuUnsupportedError } from './gpu/renderer';
import { buildPreset, type BodySet, type PresetId } from './presets';

type Mode = 'gpu-tiled' | 'cpu-barnes-hut' | 'cpu-naive';

/** CPU solvers are single-threaded JS; these keep the page interactive. */
const MODE_LIMITS: Record<Mode, number> = {
  'gpu-tiled': MAX_BODIES,
  'cpu-barnes-hut': 32768,
  'cpu-naive': 4096,
};

const ATTRACTOR_LIFETIME_MS = 2600;

const el = <T extends HTMLElement>(id: string): T => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node as T;
};

const canvas = el<HTMLCanvasElement>('gpu-canvas');

// ---------------------------------------------------------------------------
// formatting helpers
// ---------------------------------------------------------------------------

const formatInt = (n: number): string => Math.round(n).toLocaleString('en-US');

function formatMs(ms: number | null): string {
  if (ms === null || !Number.isFinite(ms)) return '—';
  if (ms < 1) return `${ms.toFixed(2)} ms`;
  if (ms < 100) return `${ms.toFixed(1)} ms`;
  return `${Math.round(ms)} ms`;
}

function formatRate(perSecond: number): string {
  if (!Number.isFinite(perSecond) || perSecond <= 0) return '—';
  const units: Array<[number, string]> = [
    [1e12, 'T'],
    [1e9, 'G'],
    [1e6, 'M'],
    [1e3, 'K'],
  ];
  for (const [scale, suffix] of units) {
    if (perSecond >= scale) return `${(perSecond / scale).toFixed(2)}${suffix}`;
  }
  return perSecond.toFixed(0);
}

// ---------------------------------------------------------------------------
// bootstrap
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  let renderer: Renderer;
  try {
    renderer = await Renderer.create(canvas);
  } catch (error) {
    const overlay = el('unsupported');
    overlay.hidden = false;
    el('unsupported-reason').textContent =
      error instanceof WebGpuUnsupportedError || error instanceof Error ? error.message : String(error);
    return;
  }

  const camera = new OrbitCamera(canvas);

  const state = {
    mode: 'gpu-tiled' as Mode,
    preset: 'galaxy' as PresetId,
    requestedExponent: 15,
    g: 1,
    dt: 0.004,
    // Must match SEED_SOFTENING in presets.ts: the initial orbital velocities are
    // derived from the softened potential.
    softening: 0.35,
    theta: 0.75,
    trail: 0.82,
    paused: false,
  };

  const solvers: Record<'cpu-naive' | 'cpu-barnes-hut', CpuSolver> = {
    'cpu-naive': new NaiveSolver(),
    'cpu-barnes-hut': new BarnesHutSolver(),
  };

  let bodies: BodySet = buildPreset(state.preset, 1 << state.requestedExponent);
  let totalMass = 0;
  // CPU solvers own their own copy of state; the GPU keeps its own in storage buffers.
  let cpuPositions = new Float32Array(0);
  let cpuVelocities = new Float32Array(0);

  let attractor: [number, number, number, number] = [0, 0, 0, 0];
  let attractorMass = 0;
  let attractorExpiry = 0;

  let cpuStepMs: number | null = null;
  const frameTimes: number[] = [];
  let benchmarking = false;

  const effectiveCount = (): number =>
    Math.min(1 << state.requestedExponent, MODE_LIMITS[state.mode]);

  function reseed(): void {
    bodies = buildPreset(state.preset, effectiveCount());
    state.dt = bodies.suggestedDt;
    el<HTMLInputElement>('ctrl-dt').value = String(state.dt);
    el<HTMLOutputElement>('out-dt').value = state.dt.toFixed(4);

    totalMass = 0;
    for (let i = 0; i < bodies.count; i++) totalMass += bodies.positions[i * 4 + 3];

    renderer.setBodies(bodies);

    if (state.mode !== 'gpu-tiled') {
      cpuPositions = bodies.positions.slice();
      cpuVelocities = bodies.velocities.slice();
    }

    camera.distance = bodies.extent * 1.9;
    camera.target = [0, 0, 0];
    attractorMass = 0;
    attractor = [0, 0, 0, 0];
    syncLabels();
  }

  function syncLabels(): void {
    el('body-count-label').textContent = formatInt(bodies.count);
    el('backend-label').textContent =
      state.mode === 'gpu-tiled' ? `WebGPU compute · ${renderer.adapterInfo}` : solvers[state.mode].label;
    el<HTMLOutputElement>('out-count').value = formatInt(effectiveCount());
    el<HTMLOutputElement>('out-g').value = state.g.toFixed(2);
    el<HTMLOutputElement>('out-soft').value = state.softening.toFixed(2);
    el<HTMLOutputElement>('out-theta').value = state.theta.toFixed(2);
    el<HTMLOutputElement>('out-trail').value = state.trail.toFixed(2);
  }

  // --- controls ------------------------------------------------------------

  const modeSelect = el<HTMLSelectElement>('ctrl-mode');
  modeSelect.addEventListener('change', () => {
    state.mode = modeSelect.value as Mode;
    // Re-seeding on solver change means every solver runs the same initial
    // conditions, which is what makes the comparison meaningful.
    reseed();
  });

  const presetSelect = el<HTMLSelectElement>('ctrl-preset');
  presetSelect.addEventListener('change', () => {
    state.preset = presetSelect.value as PresetId;
    reseed();
  });

  const countSlider = el<HTMLInputElement>('ctrl-count');
  countSlider.addEventListener('input', () => {
    state.requestedExponent = Number(countSlider.value);
    el<HTMLOutputElement>('out-count').value = formatInt(effectiveCount());
  });
  countSlider.addEventListener('change', reseed);

  const bind = (id: string, outId: string, apply: (value: number) => void, digits = 2): void => {
    const input = el<HTMLInputElement>(id);
    input.addEventListener('input', () => {
      const value = Number(input.value);
      apply(value);
      el<HTMLOutputElement>(outId).value = value.toFixed(digits);
    });
  };

  bind('ctrl-g', 'out-g', (v) => (state.g = v));
  bind('ctrl-dt', 'out-dt', (v) => (state.dt = v), 4);
  bind('ctrl-soft', 'out-soft', (v) => (state.softening = v));
  bind('ctrl-theta', 'out-theta', (v) => (state.theta = v));
  bind('ctrl-trail', 'out-trail', (v) => (state.trail = v));

  const pauseButton = el<HTMLButtonElement>('ctrl-pause');
  pauseButton.addEventListener('click', () => {
    state.paused = !state.paused;
    pauseButton.textContent = state.paused ? 'resume' : 'pause';
  });

  el('ctrl-reset').addEventListener('click', reseed);

  const panel = el('controls');
  const panelToggle = el<HTMLButtonElement>('panel-toggle');
  panelToggle.addEventListener('click', () => {
    const collapsed = panel.classList.toggle('collapsed');
    panelToggle.setAttribute('aria-expanded', String(!collapsed));
  });

  // Click (as opposed to drag) drops a temporary heavy attractor.
  let pointerDownAt = { x: 0, y: 0, time: 0 };
  canvas.addEventListener('pointerdown', (e) => {
    pointerDownAt = { x: e.clientX, y: e.clientY, time: performance.now() };
  });
  canvas.addEventListener('pointerup', (e) => {
    const moved = Math.hypot(e.clientX - pointerDownAt.x, e.clientY - pointerDownAt.y);
    if (moved > 5 || performance.now() - pointerDownAt.time > 600) return;
    const rect = canvas.getBoundingClientRect();
    const world = camera.screenToTargetPlane(
      e.clientX - rect.left,
      e.clientY - rect.top,
      rect.width,
      rect.height,
    );
    attractor = [world[0], world[1], world[2], 0];
    attractorMass = totalMass * 0.25;
    attractorExpiry = performance.now() + ATTRACTOR_LIFETIME_MS;
  });

  // --- benchmark -----------------------------------------------------------

  el('bench-close').addEventListener('click', () => {
    el('bench').hidden = true;
  });
  el('ctrl-bench').addEventListener('click', () => void runBenchmark());

  async function timeGpu(count: number, steps: number): Promise<number> {
    const scenario = buildPreset(state.preset, count);
    renderer.setBodies(scenario);
    const params = {
      dt: state.dt,
      g: state.g,
      softening: state.softening,
      damping: 1,
      attractor: [0, 0, 0, 0] as [number, number, number, number],
    };
    renderer.step(params, 4);
    await renderer.waitForGpu();

    const start = performance.now();
    renderer.step(params, steps);
    await renderer.waitForGpu();
    return (performance.now() - start) / steps;
  }

  function timeCpu(solver: CpuSolver, count: number, steps: number): number {
    const scenario = buildPreset(state.preset, count);
    const positions = scenario.positions.slice();
    const velocities = scenario.velocities.slice();
    const params: CpuStepParams = {
      dt: state.dt,
      g: state.g,
      softening: state.softening,
      damping: 1,
      theta: state.theta,
      attractor: [0, 0, 0, 0],
    };
    solver.step(positions, velocities, scenario.count, params);

    const start = performance.now();
    for (let i = 0; i < steps; i++) {
      solver.step(positions, velocities, scenario.count, params);
    }
    return (performance.now() - start) / steps;
  }

  async function runBenchmark(): Promise<void> {
    if (benchmarking) return;
    benchmarking = true;
    const wasPaused = state.paused;
    state.paused = true;

    const panelEl = el('bench');
    const tbody = el('bench-table').querySelector('tbody')!;
    const note = el('bench-note');
    panelEl.hidden = false;
    tbody.innerHTML = '<tr><td colspan="4">measuring…</td></tr>';
    note.textContent = '';

    // Every solver runs the same body count so the numbers are comparable.
    const fairCount = 4096;
    const scaleCount = Math.min(1 << state.requestedExponent, MAX_BODIES);

    await new Promise((resolve) => requestAnimationFrame(resolve));

    const rows: Array<{ label: string; count: number; ms: number }> = [];
    rows.push({ label: 'CPU naive', count: fairCount, ms: timeCpu(solvers['cpu-naive'], fairCount, 6) });
    rows.push({
      label: `CPU Barnes-Hut θ=${state.theta.toFixed(2)}`,
      count: fairCount,
      ms: timeCpu(solvers['cpu-barnes-hut'], fairCount, 6),
    });
    rows.push({ label: 'GPU tiled', count: fairCount, ms: await timeGpu(fairCount, 60) });
    if (scaleCount > fairCount) {
      rows.push({ label: 'GPU tiled', count: scaleCount, ms: await timeGpu(scaleCount, 30) });
    }

    const baseline = rows[0].ms;
    const best = Math.min(...rows.map((r) => r.ms));
    tbody.innerHTML = rows
      .map(
        (row) => `<tr class="${row.ms === best ? 'winner' : ''}">
          <td>${row.label}</td>
          <td>${formatInt(row.count)}</td>
          <td>${row.ms < 1 ? row.ms.toFixed(3) : row.ms.toFixed(2)}</td>
          <td>${(baseline / row.ms).toFixed(1)}×</td>
        </tr>`,
      )
      .join('');

    const gpuFair = rows.find((r) => r.label === 'GPU tiled' && r.count === fairCount);
    note.textContent = gpuFair
      ? `Speedup is relative to direct summation on the CPU at ${formatInt(fairCount)} bodies. ` +
        `The GPU still evaluates all ${formatInt(fairCount)}² pairs — it is ${(baseline / gpuFair.ms).toFixed(0)}× ` +
        `faster on identical work, whereas Barnes-Hut wins by doing asymptotically less work.`
      : '';

    reseed();
    state.paused = wasPaused;
    benchmarking = false;
  }

  // --- frame loop ----------------------------------------------------------

  function updateStats(now: number): void {
    frameTimes.push(now);
    while (frameTimes.length > 0 && now - frameTimes[0] > 1000) frameTimes.shift();
    const fps = frameTimes.length > 1
      ? (frameTimes.length - 1) / ((now - frameTimes[0]) / 1000)
      : 0;

    const isGpu = state.mode === 'gpu-tiled';
    const stepMs = isGpu ? renderer.computeMs : cpuStepMs;

    el('stat-fps').textContent = fps > 0 ? fps.toFixed(0) : '—';
    el('stat-step').textContent = formatMs(stepMs);
    el('stat-bodies').textContent = formatInt(bodies.count);

    const pairs = isGpu
      ? bodies.count * bodies.count
      : solvers[state.mode as 'cpu-naive' | 'cpu-barnes-hut'].lastInteractions * 2;
    el('stat-rate').textContent = stepMs ? formatRate(pairs / (stepMs / 1000)) : '—';

    if (isGpu) {
      el('stat-note').textContent =
        `tiled compute, 64-wide workgroups\n${formatRate(pairs)} pair forces per step` +
        (renderer.hasTimestamps ? '' : '\nno timestamp-query: step time unavailable');
    } else {
      const solver = solvers[state.mode as 'cpu-naive' | 'cpu-barnes-hut'];
      const naivePairs = (bodies.count * (bodies.count - 1)) / 2;
      const percent = ((solver.lastInteractions / naivePairs) * 100).toFixed(1);
      el('stat-note').textContent =
        state.mode === 'cpu-barnes-hut'
          ? `${formatInt(solver.lastNodes)} octree nodes\n${formatInt(solver.lastInteractions)} force evals = ${percent}% of naive`
          : `${formatInt(solver.lastInteractions)} pair forces per step\ncapped at ${formatInt(MODE_LIMITS['cpu-naive'])} bodies on CPU`;
    }
  }

  function frame(): void {
    const now = performance.now();

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    renderer.resize(width, height);
    camera.setAspect(width / height);

    // Attractor mass eases out so bodies are not left with a discontinuity.
    if (attractorMass > 0) {
      const remaining = (attractorExpiry - now) / ATTRACTOR_LIFETIME_MS;
      attractor[3] = remaining > 0 ? attractorMass * Math.min(1, remaining * 1.6) : 0;
      if (remaining <= 0) attractorMass = 0;
    }

    if (!state.paused && !benchmarking) {
      if (state.mode === 'gpu-tiled') {
        renderer.step(
          {
            dt: state.dt,
            g: state.g,
            softening: state.softening,
            damping: 1,
            attractor,
          },
          1,
        );
      } else {
        const solver = solvers[state.mode];
        const start = performance.now();
        solver.step(cpuPositions, cpuVelocities, bodies.count, {
          dt: state.dt,
          g: state.g,
          softening: state.softening,
          damping: 1,
          theta: state.theta,
          attractor,
        });
        const measured = performance.now() - start;
        cpuStepMs = cpuStepMs === null ? measured : cpuStepMs * 0.85 + measured * 0.15;
        renderer.uploadState(cpuPositions, cpuVelocities, bodies.count);
      }
    }

    // Sprite radius shrinks as the body count grows so dense scenes stay readable.
    const focalPx = (0.5 * height) / Math.tan(camera.fovY / 2);
    const densityScale = Math.min(4, Math.max(0.35, Math.cbrt(16384 / Math.max(bodies.count, 1))));
    const pointScale = bodies.extent * 0.0022 * densityScale * focalPx;
    const brightness = Math.min(1.8, Math.max(0.08, 0.55 * Math.pow(16384 / Math.max(bodies.count, 1), 0.28)));

    renderer.render({
      viewProjection: camera.viewProjection(),
      pointScale,
      speedScale: 1 / (bodies.referenceSpeed * 1.25),
      brightness,
      fade: state.paused ? 1 : state.trail,
      exposure: 1.15,
    });

    updateStats(now);
    requestAnimationFrame(frame);
  }

  // Dev-only probe: verifies that bodies are finite, sitting inside the view
  // frustum, and actually lighting up pixels. Useful because WebGPU canvases do
  // not always survive automated screenshots.
  if (import.meta.env.DEV) {
    Object.assign(window, {
      __nbody: async () => {
        const sample = Math.min(bodies.count, 8192);
        const positions = await renderer.readPositions(sample);
        const viewProj = camera.viewProjection();

        let finite = 0;
        let onScreen = 0;
        const radii: number[] = [];
        for (let i = 0; i < sample; i++) {
          const x = positions[i * 4];
          const y = positions[i * 4 + 1];
          const z = positions[i * 4 + 2];
          if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
          finite++;
          radii.push(Math.hypot(x, y, z));

          const cw = viewProj[3] * x + viewProj[7] * y + viewProj[11] * z + viewProj[15];
          const cx = viewProj[0] * x + viewProj[4] * y + viewProj[8] * z + viewProj[12];
          const cy = viewProj[1] * x + viewProj[5] * y + viewProj[9] * z + viewProj[13];
          if (cw > 0 && Math.abs(cx / cw) <= 1 && Math.abs(cy / cw) <= 1) onScreen++;
        }

        radii.sort((a, b) => a - b);
        const percentile = (p: number) =>
          radii.length ? Number(radii[Math.floor((radii.length - 1) * p)].toFixed(1)) : 0;

        return {
          mode: state.mode,
          preset: state.preset,
          bodies: bodies.count,
          finiteFraction: finite / sample,
          onScreenFraction: onScreen / sample,
          // A healthy disk keeps p99 near the seeded extent; a runaway p99 means
          // bodies are being ejected by unresolved close encounters.
          radius: { p50: percentile(0.5), p90: percentile(0.9), p99: percentile(0.99), max: percentile(1) },
          extent: bodies.extent,
          cameraDistance: Number(camera.distance.toFixed(1)),
          pixels: await renderer.probePixels(),
        };
      },
    });
  }

  countSlider.value = String(state.requestedExponent);
  reseed();
  requestAnimationFrame(frame);
}

void main();
