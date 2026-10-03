import './style.css';
import { FaceLandmarker, FilesetResolver, type FaceLandmarkerResult } from '@mediapipe/tasks-vision';

const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const WASM_ROOT = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
const INFERENCE_INTERVAL_MS = 1000 / 12;

const SMILE_ENTER = 0.50;
const SMILE_EXIT = 0.40;
const LAUGH_ENTER_SMILE = 0.68;
const LAUGH_ENTER_JAW = 0.50;
const LAUGH_EXIT_SMILE = 0.56;
const LAUGH_EXIT_JAW = 0.35;
const STATE_HOLD_MS = 180;
const FIREWORK_COOLDOWN_MS = 900;
const MAX_PARTICLES = 520;

type UiState = 'idle' | 'loading' | 'running' | 'error';
type InteractionState = 'IDLE' | 'SMILE' | 'LAUGH';
type ParticleKind = 'rain' | 'firework';

interface HeadBounds {
  cx: number;
  cy: number;
  rx: number;
  ry: number;
}

interface FeatureSnapshot {
  faceDetected: boolean;
  smileScore: number;
  jawOpen: number;
  inferenceMs: number;
  lastUpdatedAt: number;
  headBounds: HeadBounds | null;
}

interface Particle {
  kind: ParticleKind;
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  life: number;
  maxLife: number;
  hue: number;
  alpha: number;
}

const app = document.querySelector<HTMLDivElement>('#app');
if (!app) throw new Error('App root not found');

app.innerHTML = `
  <main class="shell">
    <section class="hero">
      <div class="eyebrow">PART 2 · VIBE CODING / SPIKE 02</div>
      <h1>雨幕与焰火</h1>
      <p class="subtitle">微笑让雨落下，大笑让焰火绽放；先用真实表情数据验证互动状态。</p>
    </section>

    <section class="stage-card">
      <div class="stage-wrap">
        <video id="camera" autoplay muted playsinline></video>
        <canvas id="overlay"></canvas>
        <div class="stage-shade"></div>
        <div id="stage-message" class="stage-message">
          <div class="message-icon">◌</div>
          <strong>点击开始体验</strong>
          <span>本阶段只读取摄像头和面部特征，不会保存画面。</span>
        </div>
      </div>
      <div class="controls">
        <button id="start-button" class="primary-button">开始体验</button>
        <button id="reset-button" class="ghost-button" disabled>重置</button>
      </div>
    </section>

    <section class="hud-card">
      <div class="hud-header">
        <div>
          <div class="section-label">DEBUG HUD</div>
          <h2>感知状态</h2>
        </div>
        <span id="status-pill" class="status-pill">未启动</span>
      </div>
      <div class="metrics">
        <div class="metric"><span>Face detected</span><strong id="face-value">—</strong></div>
        <div class="metric"><span>Smile score</span><strong id="smile-value">0.00</strong></div>
        <div class="metric"><span>Jaw open</span><strong id="jaw-value">0.00</strong></div>
        <div class="metric"><span>Inference</span><strong id="inference-value">—</strong></div>
        <div class="metric"><span>Render FPS</span><strong id="fps-value">0</strong></div>
        <div class="metric"><span>Current effect</span><strong id="effect-value">IDLE</strong></div>
      </div>
      <p id="hint" class="hint">技术验证目标：确认表情状态可以稳定驱动不同效果。</p>
    </section>

    <footer class="footer-note">Local prototype · Camera processing stays in the browser</footer>
  </main>
`;

const video = document.querySelector<HTMLVideoElement>('#camera')!;
const canvas = document.querySelector<HTMLCanvasElement>('#overlay')!;
const ctx = canvas.getContext('2d');
const startButton = document.querySelector<HTMLButtonElement>('#start-button')!;
const resetButton = document.querySelector<HTMLButtonElement>('#reset-button')!;
const stageMessage = document.querySelector<HTMLDivElement>('#stage-message')!;
const statusPill = document.querySelector<HTMLSpanElement>('#status-pill')!;
const faceValue = document.querySelector<HTMLElement>('#face-value')!;
const smileValue = document.querySelector<HTMLElement>('#smile-value')!;
const jawValue = document.querySelector<HTMLElement>('#jaw-value')!;
const inferenceValue = document.querySelector<HTMLElement>('#inference-value')!;
const fpsValue = document.querySelector<HTMLElement>('#fps-value')!;
const effectValue = document.querySelector<HTMLElement>('#effect-value')!;
const hint = document.querySelector<HTMLElement>('#hint')!;

let faceLandmarker: FaceLandmarker | null = null;
let stream: MediaStream | null = null;
let uiState: UiState = 'idle';
let interactionState: InteractionState = 'IDLE';
let candidateState: InteractionState = 'IDLE';
let candidateSince = 0;
let lastFireworkAt = -Infinity;
let lastInferenceAt = 0;
let smoothedSmile = 0;
let smoothedJaw = 0;
let latestLandmarks: FaceLandmarkerResult['faceLandmarks'][number] | null = null;
let latestFeatures: FeatureSnapshot = {
  faceDetected: false,
  smileScore: 0,
  jawOpen: 0,
  inferenceMs: 0,
  lastUpdatedAt: 0,
  headBounds: null,
};
let frameCount = 0;
let lastFpsAt = performance.now();
let renderFps = 0;
let previousRenderAt = performance.now();
let rainAccumulator = 0;
const particles: Particle[] = [];

function setUiState(next: UiState, message?: string): void {
  uiState = next;
  const labels: Record<UiState, string> = {
    idle: '未启动',
    loading: '加载中',
    running: '运行中',
    error: '需要处理',
  };
  statusPill.textContent = labels[next];
  statusPill.dataset.state = next;
  if (message) hint.textContent = message;
}

function resizeCanvas(): void {
  const rect = video.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  ctx?.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function updateHud(): void {
  faceValue.textContent = latestFeatures.faceDetected ? 'true' : 'false';
  faceValue.dataset.active = String(latestFeatures.faceDetected);
  smileValue.textContent = latestFeatures.smileScore.toFixed(2);
  jawValue.textContent = latestFeatures.jawOpen.toFixed(2);
  inferenceValue.textContent = latestFeatures.lastUpdatedAt ? `${latestFeatures.inferenceMs.toFixed(1)} ms` : '—';
  fpsValue.textContent = String(renderFps);
  effectValue.textContent = interactionState;
  effectValue.dataset.effect = interactionState.toLowerCase();
}

function getBlendshapeScore(result: FaceLandmarkerResult, name: string): number {
  const categories = result.faceBlendshapes?.[0]?.categories ?? [];
  return categories.find((category) => category.categoryName === name)?.score ?? 0;
}

function getHeadBounds(landmarks: FaceLandmarkerResult['faceLandmarks'][number], rect: DOMRect): HeadBounds {
  let minX = 1;
  let maxX = 0;
  let minY = 1;
  let maxY = 0;
  for (const point of landmarks) {
    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y);
    maxY = Math.max(maxY, point.y);
  }
  const width = (maxX - minX) * rect.width;
  const height = (maxY - minY) * rect.height;
  return {
    cx: ((minX + maxX) / 2) * rect.width,
    cy: ((minY + maxY) / 2) * rect.height,
    rx: Math.max(48, width * 0.62),
    ry: Math.max(60, height * 0.68),
  };
}

function readFeatures(result: FaceLandmarkerResult, inferenceMs: number, rect: DOMRect): FeatureSnapshot {
  const hasFace = result.faceLandmarks.length > 0;
  const leftSmile = getBlendshapeScore(result, 'mouthSmileLeft');
  const rightSmile = getBlendshapeScore(result, 'mouthSmileRight');
  const smileScore = (leftSmile + rightSmile) / 2;
  const jawOpen = getBlendshapeScore(result, 'jawOpen');
  smoothedSmile = smoothedSmile * 0.72 + smileScore * 0.28;
  smoothedJaw = smoothedJaw * 0.72 + jawOpen * 0.28;
  return {
    faceDetected: hasFace,
    smileScore: smoothedSmile,
    jawOpen: smoothedJaw,
    inferenceMs,
    lastUpdatedAt: performance.now(),
    headBounds: hasFace ? getHeadBounds(result.faceLandmarks[0], rect) : null,
  };
}

function desiredInteractionState(): InteractionState {
  if (!latestFeatures.faceDetected) return 'IDLE';

  // Hysteresis: once in LAUGH or SMILE, use softer exit thresholds to avoid flicker.
  if (interactionState === 'LAUGH' && smoothedSmile >= LAUGH_EXIT_SMILE && smoothedJaw >= LAUGH_EXIT_JAW) return 'LAUGH';
  if (interactionState === 'SMILE' && smoothedSmile >= SMILE_EXIT && smoothedJaw <= 0.50) return 'SMILE';

  if (smoothedSmile >= LAUGH_ENTER_SMILE && smoothedJaw >= LAUGH_ENTER_JAW) return 'LAUGH';
  if (smoothedSmile >= SMILE_ENTER && smoothedJaw <= 0.38) return 'SMILE';
  return 'IDLE';
}

function transitionInteraction(next: InteractionState, now: number): void {
  if (next === interactionState) return;
  const previous = interactionState;
  interactionState = next;
  updateHud();

  if (next === 'LAUGH' && now - lastFireworkAt >= FIREWORK_COOLDOWN_MS) {
    spawnFirework();
    lastFireworkAt = now;
  }

  const messages: Record<InteractionState, string> = {
    IDLE: '状态稳定：保持自然表情，效果会逐渐停下。',
    SMILE: '检测到微笑：雨幕已启动。',
    LAUGH: '检测到大笑：焰火已触发，粒子会与头部边界互动。',
  };
  hint.textContent = messages[next];
  console.info(`[interaction] ${previous} → ${next}`, { smile: smoothedSmile.toFixed(2), jaw: smoothedJaw.toFixed(2) });
}

function updateInteraction(now: number): void {
  const desired = desiredInteractionState();
  if (desired !== candidateState) {
    candidateState = desired;
    candidateSince = now;
  }
  if (candidateState !== interactionState && now - candidateSince >= STATE_HOLD_MS) {
    transitionInteraction(candidateState, now);
  }
}

function drawDebugFace(): void {
  if (!ctx || !latestLandmarks) return;
  const rect = video.getBoundingClientRect();
  ctx.save();
  ctx.fillStyle = 'rgba(141, 255, 207, 0.72)';
  for (const point of latestLandmarks) {
    // Canvas is mirrored with the video via CSS, so source coordinates stay unflipped here.
    const x = point.x * rect.width;
    const y = point.y * rect.height;
    ctx.beginPath();
    ctx.arc(x, y, 1.2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function pushParticle(particle: Particle): void {
  if (particles.length >= MAX_PARTICLES) particles.shift();
  particles.push(particle);
}

function spawnRainDrop(width: number): void {
  pushParticle({
    kind: 'rain',
    x: Math.random() * width,
    y: -12,
    vx: -10 + Math.random() * 20,
    vy: 250 + Math.random() * 180,
    size: 7 + Math.random() * 10,
    life: 2.2,
    maxLife: 2.2,
    hue: 188 + Math.random() * 35,
    alpha: 0.28 + Math.random() * 0.34,
  });
}

function spawnFirework(): void {
  const rect = video.getBoundingClientRect();
  const bounds = latestFeatures.headBounds;
  const originX = bounds?.cx ?? rect.width * 0.5;
  const originY = bounds ? Math.max(48, bounds.cy - bounds.ry * 1.35) : rect.height * 0.35;
  const count = 84;
  const hue = 28 + Math.random() * 100;
  for (let index = 0; index < count; index += 1) {
    const angle = (Math.PI * 2 * index) / count + (Math.random() - 0.5) * 0.16;
    const speed = 100 + Math.random() * 260;
    pushParticle({
      kind: 'firework',
      x: originX,
      y: originY,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      size: 2 + Math.random() * 2.6,
      life: 0.75 + Math.random() * 0.55,
      maxLife: 1.3,
      hue: hue + (Math.random() - 0.5) * 35,
      alpha: 0.8 + Math.random() * 0.2,
    });
  }
}

function collideWithHead(particle: Particle, bounds: HeadBounds): void {
  const dx = (particle.x - bounds.cx) / bounds.rx;
  const dy = (particle.y - bounds.cy) / bounds.ry;
  const distance = Math.sqrt(dx * dx + dy * dy);
  if (distance >= 1) return;

  const safeDistance = Math.max(distance, 0.001);
  const nx = dx / safeDistance;
  const ny = dy / safeDistance;
  particle.x = bounds.cx + nx * bounds.rx * 1.02;
  particle.y = bounds.cy + ny * bounds.ry * 1.02;
  const velocityAlongNormal = particle.vx * nx + particle.vy * ny;
  if (velocityAlongNormal < 0) {
    particle.vx -= 1.65 * velocityAlongNormal * nx;
    particle.vy -= 1.65 * velocityAlongNormal * ny;
    particle.vx *= 0.72;
    particle.vy *= 0.72;
  }
}

function updateParticles(deltaSeconds: number, rect: DOMRect): void {
  if (interactionState === 'SMILE') {
    rainAccumulator += deltaSeconds * 155;
    while (rainAccumulator >= 1) {
      spawnRainDrop(rect.width);
      rainAccumulator -= 1;
    }
  } else {
    rainAccumulator = Math.min(rainAccumulator, 1);
  }

  for (let index = particles.length - 1; index >= 0; index -= 1) {
    const particle = particles[index];
    particle.life -= deltaSeconds;
    if (particle.life <= 0 || particle.y > rect.height + 40 || particle.x < -80 || particle.x > rect.width + 80) {
      particles.splice(index, 1);
      continue;
    }

    if (particle.kind === 'rain') {
      particle.vy += 60 * deltaSeconds;
    } else {
      particle.vy += 300 * deltaSeconds;
      if (latestFeatures.headBounds) collideWithHead(particle, latestFeatures.headBounds);
    }
    particle.x += particle.vx * deltaSeconds;
    particle.y += particle.vy * deltaSeconds;
  }
}

function drawParticles(): void {
  if (!ctx) return;
  ctx.save();
  for (const particle of particles) {
    const lifeRatio = Math.max(0, particle.life / particle.maxLife);
    ctx.globalAlpha = particle.alpha * Math.min(1, lifeRatio * 1.8);
    ctx.strokeStyle = `hsla(${particle.hue}, 100%, 78%, 1)`;
    ctx.fillStyle = `hsla(${particle.hue}, 100%, 70%, 1)`;
    if (particle.kind === 'rain') {
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(particle.x, particle.y);
      ctx.lineTo(particle.x - particle.vx * 0.012, particle.y - particle.vy * 0.022);
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.arc(particle.x, particle.y, particle.size * (0.7 + lifeRatio * 0.4), 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
}

function render(now: number): void {
  frameCount += 1;
  if (now - lastFpsAt >= 1000) {
    renderFps = frameCount;
    frameCount = 0;
    lastFpsAt = now;
  }

  const rect = video.getBoundingClientRect();
  if (canvas.width !== Math.floor(rect.width * Math.min(window.devicePixelRatio || 1, 2))) resizeCanvas();
  ctx?.clearRect(0, 0, rect.width, rect.height);

  if (faceLandmarker && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && now - lastInferenceAt >= INFERENCE_INTERVAL_MS) {
    lastInferenceAt = now;
    const inferenceStart = performance.now();
    const result = faceLandmarker.detectForVideo(video, now);
    latestLandmarks = result.faceLandmarks[0] ?? null;
    latestFeatures = readFeatures(result, performance.now() - inferenceStart, rect);
    updateInteraction(now);
    updateHud();
  }

  const deltaSeconds = Math.min(0.05, (now - previousRenderAt) / 1000);
  previousRenderAt = now;
  updateParticles(deltaSeconds, rect);
  drawParticles();
  drawDebugFace();
  requestAnimationFrame(render);
}

async function createLandmarker(): Promise<FaceLandmarker> {
  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
  return FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

async function startExperience(): Promise<void> {
  if (uiState === 'loading' || uiState === 'running') return;
  setUiState('loading', '正在请求摄像头权限并加载 Face Landmarker…');
  startButton.disabled = true;

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    resizeCanvas();
    faceLandmarker = await createLandmarker();
    stageMessage.classList.add('hidden');
    resetButton.disabled = false;
    setUiState('running', '请自然微笑、露齿大笑或张嘴，观察效果状态如何区分。');
  } catch (error) {
    console.error(error);
    const message = error instanceof DOMException && error.name === 'NotAllowedError'
      ? '摄像头权限被拒绝，请在浏览器地址栏中允许摄像头后重试。'
      : '初始化失败。请确认使用 HTTPS 或 localhost，并检查摄像头和网络连接。';
    setUiState('error', message);
    startButton.disabled = false;
    stageMessage.classList.remove('hidden');
    stageMessage.querySelector('strong')!.textContent = '无法开始';
    stageMessage.querySelector('span')!.textContent = message;
  }
}

function resetExperience(): void {
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  faceLandmarker?.close();
  faceLandmarker = null;
  latestLandmarks = null;
  particles.length = 0;
  smoothedSmile = 0;
  smoothedJaw = 0;
  interactionState = 'IDLE';
  candidateState = 'IDLE';
  previousRenderAt = performance.now();
  latestFeatures = { faceDetected: false, smileScore: 0, jawOpen: 0, inferenceMs: 0, lastUpdatedAt: 0, headBounds: null };
  updateHud();
  stageMessage.classList.remove('hidden');
  stageMessage.querySelector('strong')!.textContent = '点击开始体验';
  stageMessage.querySelector('span')!.textContent = '本阶段只读取摄像头和面部特征，不会保存画面。';
  resetButton.disabled = true;
  startButton.disabled = false;
  setUiState('idle', '技术验证目标：确认表情状态可以稳定驱动不同效果。');
}

startButton.addEventListener('click', () => void startExperience());
resetButton.addEventListener('click', resetExperience);
window.addEventListener('resize', resizeCanvas);
setUiState('idle');
requestAnimationFrame(render);
