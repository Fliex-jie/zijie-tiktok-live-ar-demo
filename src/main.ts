import './style.css';
import { FaceLandmarker, FilesetResolver, type FaceLandmarkerResult } from '@mediapipe/tasks-vision';

const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const WASM_ROOT = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
const INFERENCE_INTERVAL_MS = 1000 / 12;

type UiState = 'idle' | 'loading' | 'ready' | 'running' | 'error';

interface FeatureSnapshot {
  faceDetected: boolean;
  smileScore: number;
  jawOpen: number;
  inferenceMs: number;
  lastUpdatedAt: number;
}

const app = document.querySelector<HTMLDivElement>('#app');
if (!app) throw new Error('App root not found');

app.innerHTML = `
  <main class="shell">
    <section class="hero">
      <div class="eyebrow">PART 2 · VIBE CODING / SPIKE 01</div>
      <h1>雨幕与焰火</h1>
      <p class="subtitle">先验证摄像头与面部特征，再把微笑和大笑变成实时反馈。</p>
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
      </div>
      <p id="hint" class="hint">技术验证目标：确认摄像头、Face Landmarker 和实时指标链路可用。</p>
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
const hint = document.querySelector<HTMLElement>('#hint')!;

let faceLandmarker: FaceLandmarker | null = null;
let stream: MediaStream | null = null;
let uiState: UiState = 'idle';
let lastInferenceAt = 0;
let latestFeatures: FeatureSnapshot = {
  faceDetected: false,
  smileScore: 0,
  jawOpen: 0,
  inferenceMs: 0,
  lastUpdatedAt: 0,
};
let frameCount = 0;
let lastFpsAt = performance.now();
let renderFps = 0;

function setUiState(next: UiState, message?: string): void {
  uiState = next;
  const labels: Record<UiState, string> = {
    idle: '未启动',
    loading: '加载中',
    ready: '已就绪',
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
}

function getBlendshapeScore(result: FaceLandmarkerResult, name: string): number {
  const categories = result.faceBlendshapes?.[0]?.categories ?? [];
  return categories.find((category) => category.categoryName === name)?.score ?? 0;
}

function readFeatures(result: FaceLandmarkerResult, inferenceMs: number): FeatureSnapshot {
  const hasFace = result.faceLandmarks.length > 0;
  const leftSmile = getBlendshapeScore(result, 'mouthSmileLeft');
  const rightSmile = getBlendshapeScore(result, 'mouthSmileRight');
  return {
    faceDetected: hasFace,
    smileScore: (leftSmile + rightSmile) / 2,
    jawOpen: getBlendshapeScore(result, 'jawOpen'),
    inferenceMs,
    lastUpdatedAt: performance.now(),
  };
}

function drawDebugFace(result: FaceLandmarkerResult): void {
  if (!ctx || result.faceLandmarks.length === 0) return;
  const rect = video.getBoundingClientRect();
  const landmarks = result.faceLandmarks[0];
  ctx.save();
  ctx.strokeStyle = 'rgba(141, 255, 207, 0.75)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (const point of landmarks) {
    const x = (1 - point.x) * rect.width;
    const y = point.y * rect.height;
    ctx.moveTo(x + 1.5, y);
    ctx.arc(x, y, 1.5, 0, Math.PI * 2);
  }
  ctx.stroke();
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
    latestFeatures = readFeatures(result, performance.now() - inferenceStart);
    drawDebugFace(result);
    updateHud();
  }

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
    setUiState('running', '技术验证进行中：请对着摄像头做自然微笑和张嘴动作，观察分数变化。');
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
  latestFeatures = { faceDetected: false, smileScore: 0, jawOpen: 0, inferenceMs: 0, lastUpdatedAt: 0 };
  updateHud();
  stageMessage.classList.remove('hidden');
  stageMessage.querySelector('strong')!.textContent = '点击开始体验';
  stageMessage.querySelector('span')!.textContent = '本阶段只读取摄像头和面部特征，不会保存画面。';
  resetButton.disabled = true;
  startButton.disabled = false;
  setUiState('idle', '技术验证目标：确认摄像头、Face Landmarker 和实时指标链路可用。');
}

startButton.addEventListener('click', () => void startExperience());
resetButton.addEventListener('click', resetExperience);
window.addEventListener('resize', resizeCanvas);
setUiState('idle');
requestAnimationFrame(render);
