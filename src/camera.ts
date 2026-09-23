import {
  add,
  cross,
  lookAt,
  multiply,
  normalize,
  perspective,
  scale,
  sub,
  type Vec3,
} from './math';

const UP: Vec3 = [0, 1, 0];
const MIN_ELEVATION = -1.5533; // just shy of straight down, avoids a degenerate up vector
const MAX_ELEVATION = 1.5533;

/**
 * Orbit camera driven by pointer drag (azimuth/elevation) and wheel (distance).
 * Also unprojects screen clicks onto the plane through the orbit target, which is
 * how the "drop an attractor" interaction picks a 3D location from a 2D click.
 */
export class OrbitCamera {
  azimuth = 0.6;
  elevation = 0.45;
  distance = 80;
  target: Vec3 = [0, 0, 0];
  fovY = (50 * Math.PI) / 180;
  near = 0.1;
  far = 6000;

  private dragging = false;
  private lastX = 0;
  private lastY = 0;
  private aspect = 1;

  constructor(canvas: HTMLCanvasElement) {
    canvas.addEventListener('pointerdown', this.onPointerDown);
    window.addEventListener('pointermove', this.onPointerMove);
    window.addEventListener('pointerup', this.onPointerUp);
    canvas.addEventListener('wheel', this.onWheel, { passive: false });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  setAspect(aspect: number): void {
    this.aspect = aspect;
  }

  /** True while the pointer is being dragged, used to suppress click-to-spawn. */
  get isDragging(): boolean {
    return this.dragging;
  }

  get eye(): Vec3 {
    const cosEl = Math.cos(this.elevation);
    return add(this.target, [
      this.distance * cosEl * Math.sin(this.azimuth),
      this.distance * Math.sin(this.elevation),
      this.distance * cosEl * Math.cos(this.azimuth),
    ]);
  }

  viewProjection(): Float32Array {
    const view = lookAt(this.eye, this.target, UP);
    const proj = perspective(this.fovY, this.aspect, this.near, this.far);
    return multiply(proj, view);
  }

  /**
   * Converts a canvas-relative pixel position into a world-space point on the
   * plane that passes through the orbit target and faces the camera.
   */
  screenToTargetPlane(px: number, py: number, width: number, height: number): Vec3 {
    // Normalized device coordinates, y flipped (screen y grows downward).
    const ndcX = (px / width) * 2 - 1;
    const ndcY = 1 - (py / height) * 2;

    const eye = this.eye;
    const forward = normalize(sub(this.target, eye));
    const right = normalize(cross(forward, UP));
    const up = cross(right, forward);

    // Half-extents of the view frustum at the distance of the orbit target.
    const halfHeight = Math.tan(this.fovY / 2) * this.distance;
    const halfWidth = halfHeight * this.aspect;

    return add(
      this.target,
      add(scale(right, ndcX * halfWidth), scale(up, ndcY * halfHeight)),
    );
  }

  private onPointerDown = (e: PointerEvent): void => {
    this.dragging = true;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
  };

  private onPointerMove = (e: PointerEvent): void => {
    if (!this.dragging) return;
    const dx = e.clientX - this.lastX;
    const dy = e.clientY - this.lastY;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    this.azimuth -= dx * 0.005;
    this.elevation = Math.min(
      MAX_ELEVATION,
      Math.max(MIN_ELEVATION, this.elevation + dy * 0.005),
    );
  };

  private onPointerUp = (): void => {
    this.dragging = false;
  };

  private onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    this.distance = Math.min(3000, Math.max(1.5, this.distance * Math.exp(e.deltaY * 0.001)));
  };
}
