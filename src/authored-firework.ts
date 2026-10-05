type RawKeyframe = {
  t: number;
  s?: number[];
  e?: number[];
  h?: number;
  i?: { x?: number | number[]; y?: number | number[] };
  o?: { x?: number | number[]; y?: number | number[] };
};

type RawProperty = {
  a?: number;
  k?: number | number[] | RawKeyframe[];
};

interface RawPath {
  v?: number[][];
  i?: number[][];
  o?: number[][];
  c?: boolean;
}

interface RawShapeItem {
  ty?: string;
  nm?: string;
  it?: RawShapeItem[];
  ks?: { k?: RawPath };
  c?: RawProperty;
  o?: RawProperty;
  p?: RawProperty;
  a?: RawProperty;
  s?: RawProperty;
  r?: RawProperty;
}

interface RawLayer {
  nm?: string;
  ip?: number;
  op?: number;
  ks?: {
    p?: RawProperty;
    a?: RawProperty;
    s?: RawProperty;
    r?: RawProperty;
    o?: RawProperty;
  };
  shapes?: RawShapeItem[];
}

interface RawAnimation {
  fr?: number;
  ip?: number;
  op?: number;
  w?: number;
  h?: number;
  layers?: RawLayer[];
}

export interface AnimatedValue {
  value: number[];
  keyframes: RawKeyframe[] | null;
}

export interface AuthoredTransform {
  position: AnimatedValue;
  anchor: AnimatedValue;
  scale: AnimatedValue;
  rotation: AnimatedValue;
  opacity: AnimatedValue;
}

export interface AuthoredGroup {
  name: string;
  path: Path2D;
  fill: string;
  fillOpacity: number;
  transform: AuthoredTransform;
}

export interface AuthoredLayer {
  name: string;
  inFrame: number;
  outFrame: number;
  transform: AuthoredTransform;
  groups: AuthoredGroup[];
  physical: boolean;
}

export interface AuthoredFirework {
  width: number;
  height: number;
  frameRate: number;
  inFrame: number;
  outFrame: number;
  layers: AuthoredLayer[];
}

export interface EvaluatedTransform {
  x: number;
  y: number;
  anchorX: number;
  anchorY: number;
  scaleX: number;
  scaleY: number;
  rotation: number;
  opacity: number;
}

function asVector(value: unknown, fallback: number[]): number[] {
  if (typeof value === 'number') return [value];
  if (Array.isArray(value) && value.every((item) => typeof item === 'number')) return value as number[];
  return fallback;
}

function animatedValue(raw: RawProperty | undefined, fallback: number[]): AnimatedValue {
  if (!raw) return { value: fallback, keyframes: null };
  const source = raw.k;
  if (raw.a === 1 && Array.isArray(source) && source.length > 0 && typeof source[0] === 'object') {
    const keyframes = source as RawKeyframe[];
    return { value: asVector(keyframes[0]?.s, fallback), keyframes };
  }
  return { value: asVector(source, fallback), keyframes: null };
}

function controlValue(value: number | number[] | undefined, index: number, fallback: number): number {
  if (typeof value === 'number') return value;
  if (!Array.isArray(value) || value.length === 0) return fallback;
  return value[Math.min(index, value.length - 1)] ?? fallback;
}

function cubicBezierCoordinate(t: number, a: number, b: number): number {
  const inverse = 1 - t;
  return 3 * inverse * inverse * t * a + 3 * inverse * t * t * b + t * t * t;
}

function cubicBezierDerivative(t: number, a: number, b: number): number {
  const inverse = 1 - t;
  return 3 * inverse * inverse * a + 6 * inverse * t * (b - a) + 3 * t * t * (1 - b);
}

function cubicBezierProgress(progress: number, current: RawKeyframe, next: RawKeyframe): number {
  const x1 = controlValue(current.o?.x, 0, 0.333);
  const y1 = controlValue(current.o?.y, 0, 0);
  const x2 = controlValue(next.i?.x, 0, 0.667);
  const y2 = controlValue(next.i?.y, 0, 1);
  let t = progress;
  for (let iteration = 0; iteration < 5; iteration += 1) {
    const difference = cubicBezierCoordinate(t, x1, x2) - progress;
    const derivative = cubicBezierDerivative(t, x1, x2);
    if (Math.abs(derivative) < 0.0001) break;
    t = Math.max(0, Math.min(1, t - difference / derivative));
  }
  return cubicBezierCoordinate(t, y1, y2);
}

export function valueAtFrame(property: AnimatedValue, frame: number): number[] {
  const keyframes = property.keyframes;
  if (!keyframes || keyframes.length === 0) return property.value;
  if (frame <= keyframes[0].t) return asVector(keyframes[0].s, property.value);
  for (let index = 0; index < keyframes.length - 1; index += 1) {
    const current = keyframes[index];
    const next = keyframes[index + 1];
    if (frame > next.t) continue;
    const start = asVector(current.s, property.value);
    if (current.h === 1 || next.t <= current.t) return start;
    const end = asVector(current.e ?? next.s, start);
    const rawProgress = Math.max(0, Math.min(1, (frame - current.t) / (next.t - current.t)));
    const progress = cubicBezierProgress(rawProgress, current, next);
    const length = Math.max(start.length, end.length);
    return Array.from({ length }, (_, component) => {
      const from = start[Math.min(component, start.length - 1)] ?? 0;
      const to = end[Math.min(component, end.length - 1)] ?? from;
      return from + (to - from) * progress;
    });
  }
  return asVector(keyframes[keyframes.length - 1].s, property.value);
}

export function evaluateTransform(transform: AuthoredTransform, frame: number): EvaluatedTransform {
  const position = valueAtFrame(transform.position, frame);
  const anchor = valueAtFrame(transform.anchor, frame);
  const scale = valueAtFrame(transform.scale, frame);
  const rotation = valueAtFrame(transform.rotation, frame);
  const opacity = valueAtFrame(transform.opacity, frame);
  return {
    x: position[0] ?? 0,
    y: position[1] ?? 0,
    anchorX: anchor[0] ?? 0,
    anchorY: anchor[1] ?? 0,
    scaleX: (scale[0] ?? 100) / 100,
    scaleY: (scale[1] ?? scale[0] ?? 100) / 100,
    rotation: ((rotation[0] ?? 0) * Math.PI) / 180,
    opacity: (opacity[0] ?? 100) / 100,
  };
}

function createTransform(source: RawShapeItem | RawLayer['ks'] | undefined): AuthoredTransform {
  return {
    position: animatedValue(source?.p, [0, 0]),
    anchor: animatedValue(source?.a, [0, 0]),
    scale: animatedValue(source?.s, [100, 100]),
    rotation: animatedValue(source?.r, [0]),
    opacity: animatedValue(source?.o, [100]),
  };
}

function createPath(path: RawPath | undefined): Path2D {
  const result = new Path2D();
  const vertices = path?.v ?? [];
  const incoming = path?.i ?? [];
  const outgoing = path?.o ?? [];
  if (vertices.length === 0) return result;
  result.moveTo(vertices[0][0] ?? 0, vertices[0][1] ?? 0);
  const segmentCount = path?.c ? vertices.length : vertices.length - 1;
  for (let index = 0; index < segmentCount; index += 1) {
    const nextIndex = (index + 1) % vertices.length;
    const current = vertices[index];
    const next = vertices[nextIndex];
    const currentOutgoing = outgoing[index] ?? [0, 0];
    const nextIncoming = incoming[nextIndex] ?? [0, 0];
    result.bezierCurveTo(
      (current[0] ?? 0) + (currentOutgoing[0] ?? 0),
      (current[1] ?? 0) + (currentOutgoing[1] ?? 0),
      (next[0] ?? 0) + (nextIncoming[0] ?? 0),
      (next[1] ?? 0) + (nextIncoming[1] ?? 0),
      next[0] ?? 0,
      next[1] ?? 0,
    );
  }
  if (path?.c) result.closePath();
  return result;
}

function colorToCss(raw: RawProperty | undefined, fallback = 'rgba(255,255,255,1)'): string {
  const color = animatedValue(raw, []).value;
  if (color.length < 3) return fallback;
  const red = Math.round(Math.max(0, Math.min(1, color[0])) * 255);
  const green = Math.round(Math.max(0, Math.min(1, color[1])) * 255);
  const blue = Math.round(Math.max(0, Math.min(1, color[2])) * 255);
  const alpha = Math.max(0, Math.min(1, color[3] ?? 1));
  return `rgba(${red},${green},${blue},${alpha})`;
}

function parseLayer(raw: RawLayer): AuthoredLayer {
  const shapes = raw.shapes ?? [];
  const name = raw.nm ?? 'Layer';
  const layerFillItem = shapes.find((item) => item.ty === 'fl');
  const fallbackFill = colorToCss(layerFillItem?.c);
  const fallbackFillOpacity = (animatedValue(layerFillItem?.o, [100]).value[0] ?? 100) / 100;
  const groups: AuthoredGroup[] = [];
  for (const rawGroup of shapes) {
    if (rawGroup.ty !== 'gr') continue;
    const items = rawGroup.it ?? [];
    const pathItem = items.find((item) => item.ty === 'sh');
    const transformItem = items.find((item) => item.ty === 'tr');
    const fillItem = items.find((item) => item.ty === 'fl');
    if (!pathItem?.ks?.k || !transformItem) continue;
    groups.push({
      name: rawGroup.nm ?? `Group ${groups.length + 1}`,
      path: createPath(pathItem.ks.k),
      fill: fillItem ? colorToCss(fillItem.c, fallbackFill) : fallbackFill,
      fillOpacity: fillItem
        ? (animatedValue(fillItem.o, [100]).value[0] ?? 100) / 100
        : fallbackFillOpacity,
      transform: createTransform(transformItem),
    });
  }
  return {
    name,
    inFrame: raw.ip ?? 0,
    outFrame: raw.op ?? Number.POSITIVE_INFINITY,
    transform: createTransform(raw.ks),
    groups,
    physical: /Green Outlines|White Outlines/i.test(name),
  };
}

export async function loadAuthoredFirework(url: string): Promise<AuthoredFirework> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Unable to load authored firework: ${response.status}`);
  const raw = await response.json() as RawAnimation;
  return {
    width: raw.w ?? 900,
    height: raw.h ?? 900,
    frameRate: raw.fr ?? 30,
    inFrame: raw.ip ?? 0,
    outFrame: raw.op ?? 63,
    layers: (raw.layers ?? []).map(parseLayer),
  };
}

function applyTransform(target: CanvasRenderingContext2D, transform: EvaluatedTransform): void {
  target.translate(transform.x, transform.y);
  target.rotate(transform.rotation);
  target.scale(transform.scaleX, transform.scaleY);
  target.translate(-transform.anchorX, -transform.anchorY);
}

export function groupCompositionPosition(
  layer: AuthoredLayer,
  group: AuthoredGroup,
  frame: number,
): { x: number; y: number } {
  const layerTransform = evaluateTransform(layer.transform, frame);
  const groupTransform = evaluateTransform(group.transform, frame);
  const localX = (groupTransform.x - layerTransform.anchorX) * layerTransform.scaleX;
  const localY = (groupTransform.y - layerTransform.anchorY) * layerTransform.scaleY;
  const cosine = Math.cos(layerTransform.rotation);
  const sine = Math.sin(layerTransform.rotation);
  return {
    x: layerTransform.x + localX * cosine - localY * sine,
    y: layerTransform.y + localX * sine + localY * cosine,
  };
}

export function renderAuthoredCore(
  animation: AuthoredFirework,
  target: HTMLCanvasElement,
  frame: number,
): void {
  const context = target.getContext('2d');
  if (!context) return;
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.clearRect(0, 0, target.width, target.height);
  const outputScale = target.width / animation.width;
  context.scale(outputScale, outputScale);
  context.globalCompositeOperation = 'source-over';

  for (const layer of [...animation.layers].reverse()) {
    if (layer.physical || frame < layer.inFrame || frame > layer.outFrame) continue;
    const layerTransform = evaluateTransform(layer.transform, frame);
    context.save();
    applyTransform(context, layerTransform);
    for (const group of [...layer.groups].reverse()) {
      const groupTransform = evaluateTransform(group.transform, frame);
      context.save();
      applyTransform(context, groupTransform);
      context.globalAlpha = layerTransform.opacity * groupTransform.opacity * group.fillOpacity;
      context.fillStyle = group.fill;
      context.fill(group.path);
      context.restore();
    }
    context.restore();
  }
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.globalAlpha = 1;
}

export function drawAuthoredPhysicalGroup(
  target: CanvasRenderingContext2D,
  layer: AuthoredLayer,
  group: AuthoredGroup,
  frame: number,
  stageX: number,
  stageY: number,
  stageScale: number,
  stageRotation: number,
  alpha: number,
): void {
  if (frame < layer.inFrame || frame > layer.outFrame || alpha <= 0.001) return;
  const layerTransform = evaluateTransform(layer.transform, frame);
  const groupTransform = evaluateTransform(group.transform, frame);
  target.save();
  target.translate(stageX, stageY);
  target.rotate(stageRotation + layerTransform.rotation + groupTransform.rotation);
  target.scale(
    stageScale * layerTransform.scaleX * groupTransform.scaleX,
    stageScale * layerTransform.scaleY * groupTransform.scaleY,
  );
  target.translate(-groupTransform.anchorX, -groupTransform.anchorY);
  target.globalAlpha = alpha * layerTransform.opacity * groupTransform.opacity * group.fillOpacity;
  target.fillStyle = group.fill;
  target.shadowColor = group.fill;
  target.shadowBlur = 5 / Math.max(0.2, stageScale);
  target.fill(group.path);
  target.restore();
}
