import './style.css';
import { FaceLandmarker, FilesetResolver, type FaceLandmarkerResult } from '@mediapipe/tasks-vision';
import {
  drawAuthoredPhysicalGroup,
  groupCompositionPosition,
  renderAuthoredCore,
  type AuthoredFirework,
  type AuthoredGroup,
  type AuthoredLayer,
} from './authored-firework';
import { ProceduralFireworkSystem } from './procedural-firework';

const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const WASM_ROOT = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
const INFERENCE_INTERVAL_MS = 1000 / 10;
const RENDER_INTERVAL_MS = 1000 / 60;
const MAX_CANVAS_DPR = 1.5;
const SHOW_DEBUG_LANDMARKS = new URLSearchParams(window.location.search).has('debug');
const PREVIEW_FIREWORK = new URLSearchParams(window.location.search).get('preview') === 'firework';

// The model reports a new value every frame, so the thresholds deliberately
// use a gap between entering and leaving a state (hysteresis). This keeps a
// borderline expression from rapidly toggling the visual effect.
// A weak mouthSmile value is only supporting evidence: head tilt, a relaxed
// mouth corner and a lip press can all produce a small non-zero score. Entry
// therefore needs either one genuinely strong model signal or agreement with
// mouth geometry. Exit is intentionally faster so a rejected expression does
// not leave rain hanging over the next neutral frame.
const SMILE_ENTER = 0.15;
const SMILE_STRONG_ENTER = 0.32;
const SMILE_EXIT = 0.08;
const SMILE_STRONG_EXIT = 0.17;
const SMILE_JAW_ENTER_MAX = 0.32;
const SMILE_JAW_EXIT_MAX = 0.40;
const LAUGH_ENTER_SMILE = 0.34;
const LAUGH_ENTER_JAW = 0.30;
const LAUGH_EXIT_SMILE = 0.25;
const LAUGH_EXIT_JAW = 0.18;
const LAUGH_SMILE_TO_JAW_RATIO = 0.72;
const LIP_PRESS_ENTER = 0.085;
const CLOSED_LIP_PRESS_ENTER = 0.058;
const LIP_PRESS_MEMORY_MS = 360;
const STATE_HOLD_MS = 65;
const SMILE_HOLD_MS = 90;
const DISTANT_SMILE_HOLD_MS = 70;
const LAUGH_HOLD_MS = 35;
const SMILE_MIN_DWELL_MS = 110;
const LAUGH_MIN_DWELL_MS = 160;
const OPEN_MOUTH_ENTER = 0.46;
const OPEN_MOUTH_EXIT = 0.34;
const AUDIO_ACTIVE_THRESHOLD = 0.28;
const AUDIO_BURST_THRESHOLD = 0.58;
const AUDIO_WINDOW_MS = 1200;
const AUDIO_SUSTAIN_MS = 180;
const AUDIO_CALIBRATION_MS = 700;
const AUDIO_MIN_DYNAMIC_RANGE = 0.012;
const FIREWORK_ENTRY_COOLDOWN_MS = 260;
const FIREWORK_SEQUENCE_WINDOW_MS = 1500;
const FIREWORK_SEQUENCE_INTERVAL_MS = 620;
const FIREWORK_REPEAT_MS = 1320;
const RAIN_DROP_RATE = 18;
const RAIN_SPLASH_RATE = 6.5;
const MAX_PARTICLES = 180;
const MAX_FIREWORK_BURSTS = 1;
const MAX_COLLISION_FLASHES = 12;
const MAX_RAIN_RIPPLES = 22;
const MAX_TRACKED_FACES = 2;
const DISTANT_FACE_SCALE = 0.38;
const DISTANT_SMILE_ENTER = 0.09;
const DISTANT_SMILE_STRONG_ENTER = 0.21;
const DISTANT_SMILE_EXIT = 0.055;
const DISTANT_SMILE_STRONG_EXIT = 0.12;
const GEOMETRY_SMILE_ENTER = 0.075;
const GEOMETRY_SMILE_EXIT = 0.032;
const DISTANT_GEOMETRY_SMILE_ENTER = 0.045;
const DISTANT_GEOMETRY_SMILE_EXIT = 0.022;
const SMILE_WIDTH_ENTER = 0.385;
const SMILE_WIDTH_EXIT = 0.36;
const DISTANT_SMILE_WIDTH_ENTER = 0.355;
const DISTANT_SMILE_WIDTH_EXIT = 0.34;
const GEOMETRY_LAUGH_OPEN_ENTER = 0.058;
const GEOMETRY_LAUGH_OPEN_EXIT = 0.042;
const AUTHORED_FIREWORK_CACHE_SIZE = 360;
const AUTHORED_FIREWORK_DURATION = 1.62;

type UiState = 'idle' | 'loading' | 'running' | 'error';
type InteractionState = 'IDLE' | 'SMILE' | 'OPEN_MOUTH' | 'LAUGH';
type ParticleKind = 'rain' | 'splash' | 'ember';

interface HeadBounds {
  cx: number;
  cy: number;
  rx: number;
  ry: number;
  angle: number;
  scale: number;
}

interface MouthGeometryMetrics {
  smile: number;
  open: number;
  width: number;
}

interface FeatureSnapshot {
  faceDetected: boolean;
  smileScore: number;
  jawOpen: number;
  lipPress: number;
  geometrySmile: number;
  geometryMouthOpen: number;
  mouthWidthRatio: number;
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
  prevX?: number;
  prevY?: number;
}

interface CollisionFlash {
  x: number;
  y: number;
  radius: number;
  life: number;
  maxLife: number;
  hue: number;
}

interface RainRipple {
  x: number;
  y: number;
  radius: number;
  maxRadius: number;
  life: number;
  maxLife: number;
  alpha: number;
}

interface FireworkBurst {
  x: number;
  y: number;
  size: number;
  rotation: number;
  life: number;
  maxLife: number;
  alpha: number;
  coreCanvas: HTMLCanvasElement;
  lastCoreFrame: number;
  physicalParticles: AuthoredParticleState[];
}

interface AuthoredParticleState {
  layer: AuthoredLayer;
  group: AuthoredGroup;
  previousAuthoredX: number | null;
  previousAuthoredY: number | null;
  x: number;
  y: number;
  vx: number;
  vy: number;
  rotation: number;
  collided: boolean;
}

const app = document.querySelector<HTMLDivElement>('#app');
if (!app) throw new Error('App root not found');

app.innerHTML = `
  <main class="shell">
    <section class="hero">
      <div class="eyebrow">PART 2 · VIBE CODING / SPIKE 02</div>
      <h1>雨幕与焰火</h1>
      <p class="subtitle">微笑让雨落下，大笑让焰火绽放；用面部与声音两条感知链模拟直播互动。</p>
    </section>

    <section class="stage-card">
      <div class="stage-wrap">
        <video id="camera" autoplay muted playsinline></video>
        <div class="rain-grade" aria-hidden="true"></div>
        <video
          id="rain-footage"
          class="rain-footage"
          src="/assets/rain-overlay-candidate.mp4"
          muted
          loop
          playsinline
          preload="metadata"
          aria-hidden="true"
        ></video>
        <canvas id="overlay"></canvas>
        <div class="stage-shade"></div>
        <div id="stage-message" class="stage-message">
          <div class="message-icon">◌</div>
          <strong>点击开始体验</strong>
          <span>默认只读取摄像头；声音辅助需主动开启，所有分析都在浏览器本地完成。</span>
        </div>
      </div>
      <div class="controls">
        <button id="start-button" class="primary-button">开始体验</button>
        <button id="reset-button" class="ghost-button" disabled>重置</button>
        <button id="audio-button" class="ghost-button" disabled aria-pressed="false">启用声音辅助</button>
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
        <div class="metric"><span>Lip press</span><strong id="lip-press-value">0.00</strong></div>
        <div class="metric"><span>Inference</span><strong id="inference-value">—</strong></div>
        <div class="metric"><span>Render FPS</span><strong id="fps-value">0</strong></div>
        <div class="metric"><span>Current effect</span><strong id="effect-value">IDLE</strong></div>
        <div class="metric"><span>Audio signal</span><strong id="audio-value">OFF</strong></div>
      </div>
      <div class="audio-status-row">
        <span id="audio-status" class="audio-status" data-enabled="false">声音辅助关闭 · 不读取麦克风</span>
        <span class="audio-note">开启后仅在浏览器本地分析音量，不保存音频。</span>
      </div>
      <p id="hint" class="hint">技术验证目标：确认表情状态可以稳定驱动不同效果。</p>
    </section>

    <footer class="footer-note">Local prototype · Camera processing stays in the browser</footer>
  </main>
`;

const video = document.querySelector<HTMLVideoElement>('#camera')!;
const rainFootage = document.querySelector<HTMLVideoElement>('#rain-footage')!;
const stageWrap = document.querySelector<HTMLDivElement>('.stage-wrap')!;
const canvas = document.querySelector<HTMLCanvasElement>('#overlay')!;
const ctx = canvas.getContext('2d');
const startButton = document.querySelector<HTMLButtonElement>('#start-button')!;
const resetButton = document.querySelector<HTMLButtonElement>('#reset-button')!;
const audioButton = document.querySelector<HTMLButtonElement>('#audio-button')!;
const stageMessage = document.querySelector<HTMLDivElement>('#stage-message')!;
const statusPill = document.querySelector<HTMLSpanElement>('#status-pill')!;
const faceValue = document.querySelector<HTMLElement>('#face-value')!;
const smileValue = document.querySelector<HTMLElement>('#smile-value')!;
const jawValue = document.querySelector<HTMLElement>('#jaw-value')!;
const lipPressValue = document.querySelector<HTMLElement>('#lip-press-value')!;
const inferenceValue = document.querySelector<HTMLElement>('#inference-value')!;
const fpsValue = document.querySelector<HTMLElement>('#fps-value')!;
const effectValue = document.querySelector<HTMLElement>('#effect-value')!;
const audioValue = document.querySelector<HTMLElement>('#audio-value')!;
const audioStatus = document.querySelector<HTMLElement>('#audio-status')!;
const hint = document.querySelector<HTMLElement>('#hint')!;

let faceLandmarker: FaceLandmarker | null = null;
let stream: MediaStream | null = null;
let audioStream: MediaStream | null = null;
let audioContext: AudioContext | null = null;
let audioAnalyser: AnalyserNode | null = null;
let audioSamples: Uint8Array<ArrayBuffer> | null = null;
let audioEnabled = false;
let audioLevel = 0;
let previousAudioLevel = 0;
let audioActive = false;
let audioActiveSince = 0;
let audioStartedAt = 0;
let audioNoiseFloor = 0.005;
let audioPeak = 0.025;
let audioBurstTimes: number[] = [];
let uiState: UiState = 'idle';
let interactionState: InteractionState = 'IDLE';
let candidateState: InteractionState = 'IDLE';
let candidateSince = 0;
let stateEnteredAt = 0;
let lastFireworkAt = -Infinity;
let fireworkSequenceUntil = -Infinity;
let lastInferenceAt = 0;
let smoothedSmile = 0;
let smoothedJaw = 0;
let smoothedCheekSquint = 0;
let smoothedEyeSquint = 0;
let smoothedLipPress = 0;
let lastStrongLipPressAt = -Infinity;
let latestLandmarks: FaceLandmarkerResult['faceLandmarks'][number] | null = null;
let latestFeatures: FeatureSnapshot = {
  faceDetected: false,
  smileScore: 0,
  jawOpen: 0,
  lipPress: 0,
  geometrySmile: 0,
  geometryMouthOpen: 0,
  mouthWidthRatio: 0,
  inferenceMs: 0,
  lastUpdatedAt: 0,
  headBounds: null,
};
let frameCount = 0;
let lastFpsAt = performance.now();
let renderFps = 0;
let previousRenderAt = performance.now();
let lastVisualFrameAt = -Infinity;
let rainAccumulator = 0;
let splashAccumulator = 0;
const particles: Particle[] = [];
const collisionFlashes: CollisionFlash[] = [];
const rainRipples: RainRipple[] = [];
const fireworkBursts: FireworkBurst[] = [];
let smileEffectMix = 0;
let laughEffectMix = 0;
let rainFootagePlaying = false;
let previewInitialized = false;
let fireworkSequence = 0;
let previewCollisionHits = 0;
let authoredFirework: AuthoredFirework | null = null;
// Kept only as a dormant compatibility path while the new procedural effect
// is evaluated. The JSON is no longer fetched or rendered at runtime.
const authoredFireworkReady: Promise<AuthoredFirework | null> = Promise.resolve(null);
const proceduralFireworks = new ProceduralFireworkSystem();

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
  const dpr = Math.min(window.devicePixelRatio || 1, MAX_CANVAS_DPR);
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
  lipPressValue.textContent = latestFeatures.lipPress.toFixed(2);
  inferenceValue.textContent = latestFeatures.lastUpdatedAt ? `${latestFeatures.inferenceMs.toFixed(1)} ms` : '—';
  fpsValue.textContent = String(renderFps);
  effectValue.textContent = interactionState;
  effectValue.dataset.effect = interactionState.toLowerCase();
  audioValue.textContent = audioEnabled ? `${Math.round(audioLevel * 100)}%` : 'OFF';
  audioValue.dataset.active = String(audioActive);
  canvas.dataset.geometrySmile = latestFeatures.geometrySmile.toFixed(3);
  canvas.dataset.mouthWidth = latestFeatures.mouthWidthRatio.toFixed(3);
  canvas.dataset.faceScale = (latestFeatures.headBounds?.scale ?? 0).toFixed(3);
}

function updateAudioStatus(message: string, enabled = audioEnabled): void {
  audioStatus.textContent = message;
  audioStatus.dataset.enabled = String(enabled);
  audioButton.setAttribute('aria-pressed', String(enabled));
}

function sampleAudio(now: number): void {
  if (!audioAnalyser || !audioSamples || !audioEnabled) {
    audioActive = false;
    return;
  }

  audioAnalyser.getByteTimeDomainData(audioSamples);
  let squaredTotal = 0;
  for (const sample of audioSamples) {
    const centered = (sample - 128) / 128;
    squaredTotal += centered * centered;
  }

  // RMS is a lightweight local signal level, not speech recognition. Compare
  // it with this microphone's recent noise floor and peak instead of using a
  // device-dependent absolute volume threshold.
  const rawLevel = Math.min(1, Math.sqrt(squaredTotal / audioSamples.length) * 4.2);
  const calibrating = now - audioStartedAt < AUDIO_CALIBRATION_MS;
  if (calibrating) {
    audioNoiseFloor = audioNoiseFloor * 0.82 + rawLevel * 0.18;
    audioPeak = Math.max(rawLevel, audioNoiseFloor + AUDIO_MIN_DYNAMIC_RANGE);
    audioLevel *= 0.75;
    previousAudioLevel = 0;
    audioActive = false;
    audioActiveSince = 0;
    return;
  }

  audioNoiseFloor = rawLevel < audioNoiseFloor
    ? audioNoiseFloor * 0.84 + rawLevel * 0.16
    : audioNoiseFloor * 0.998 + rawLevel * 0.002;
  audioPeak = Math.max(rawLevel, audioPeak * 0.992, audioNoiseFloor + AUDIO_MIN_DYNAMIC_RANGE);
  const dynamicRange = Math.max(AUDIO_MIN_DYNAMIC_RANGE, audioPeak - audioNoiseFloor);
  const relativeLevel = Math.max(0, Math.min(1, (rawLevel - audioNoiseFloor) / dynamicRange));
  audioLevel = audioLevel * 0.70 + relativeLevel * 0.30;

  const wasAudioActive = audioActive;
  audioActive = audioLevel >= AUDIO_ACTIVE_THRESHOLD;
  if (audioActive && !wasAudioActive) audioActiveSince = now;
  if (!audioActive) audioActiveSince = 0;

  if (relativeLevel >= AUDIO_BURST_THRESHOLD && previousAudioLevel < AUDIO_BURST_THRESHOLD) {
    audioBurstTimes.push(now);
  }
  audioBurstTimes = audioBurstTimes.filter((timestamp) => now - timestamp <= AUDIO_WINDOW_MS);
  previousAudioLevel = relativeLevel;
}

function hasAudioLaughSupport(now: number): boolean {
  if (!audioEnabled || !audioActive) return false;
  const recentBursts = audioBurstTimes.filter((timestamp) => now - timestamp <= AUDIO_WINDOW_MS).length;
  const sustainedSound = audioActiveSince > 0 && now - audioActiveSince >= AUDIO_SUSTAIN_MS;
  return recentBursts >= 2 || sustainedSound;
}

async function enableAudioAssist(): Promise<void> {
  if (audioEnabled) {
    disableAudioAssist();
    return;
  }

  audioButton.disabled = true;
  updateAudioStatus('正在请求麦克风权限…', false);
  try {
    audioStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });
    audioContext = new AudioContext();
    if (audioContext.state === 'suspended') await audioContext.resume();
    const source = audioContext.createMediaStreamSource(audioStream);
    audioAnalyser = audioContext.createAnalyser();
    audioAnalyser.fftSize = 512;
    audioAnalyser.smoothingTimeConstant = 0.65;
    source.connect(audioAnalyser);
    audioSamples = new Uint8Array(new ArrayBuffer(audioAnalyser.fftSize));
    audioEnabled = true;
    audioLevel = 0;
    previousAudioLevel = 0;
    audioActive = false;
    audioActiveSince = 0;
    audioStartedAt = performance.now();
    audioNoiseFloor = 0.005;
    audioPeak = 0.025;
    audioBurstTimes = [];
    audioButton.textContent = '关闭声音辅助';
    audioButton.disabled = false;
    updateAudioStatus('声音辅助已开启 · 自适应环境音量', true);
    hint.textContent = '声音辅助已开启：系统会根据环境底噪和近期峰值辅助确认大笑。';
    updateHud();
  } catch (error) {
    console.error(error);
    audioStream?.getTracks().forEach((track) => track.stop());
    audioStream = null;
    audioButton.textContent = '启用声音辅助';
    audioButton.disabled = false;
    updateAudioStatus('麦克风未启用 · 继续使用纯视觉模式', false);
    hint.textContent = error instanceof DOMException && error.name === 'NotAllowedError'
      ? '麦克风权限被拒绝，摄像头识别仍可继续。'
      : '麦克风初始化失败，摄像头识别仍可继续。';
  }
}

function disableAudioAssist(): void {
  audioStream?.getTracks().forEach((track) => track.stop());
  audioStream = null;
  audioAnalyser?.disconnect();
  audioAnalyser = null;
  audioSamples = null;
  void audioContext?.close();
  audioContext = null;
  audioEnabled = false;
  audioLevel = 0;
  previousAudioLevel = 0;
  audioActive = false;
  audioActiveSince = 0;
  audioStartedAt = 0;
  audioNoiseFloor = 0.005;
  audioPeak = 0.025;
  audioBurstTimes = [];
  audioButton.textContent = '启用声音辅助';
  audioButton.disabled = uiState !== 'running';
  updateAudioStatus('声音辅助关闭 · 不读取麦克风', false);
  updateHud();
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function pseudoRandom(seed: number): number {
  return Math.abs(Math.sin(seed) * 10000) % 1;
}

function getBlendshapeScore(result: FaceLandmarkerResult, name: string, faceIndex = 0): number {
  const categories = result.faceBlendshapes?.[faceIndex]?.categories ?? [];
  return categories.find((category) => category.categoryName === name)?.score ?? 0;
}

function smoothResponsive(current: number, sample: number, riseWeight = 0.48, fallWeight = 0.30): number {
  const weight = sample > current ? riseWeight : fallWeight;
  return current + (sample - current) * weight;
}

function mapLandmarkToStage(
  point: FaceLandmarkerResult['faceLandmarks'][number][number],
  rect: DOMRect,
): { x: number; y: number } {
  const sourceWidth = Math.max(1, video.videoWidth || rect.width);
  const sourceHeight = Math.max(1, video.videoHeight || rect.height);
  const coverScale = Math.max(rect.width / sourceWidth, rect.height / sourceHeight);
  const renderedWidth = sourceWidth * coverScale;
  const renderedHeight = sourceHeight * coverScale;
  const offsetX = (rect.width - renderedWidth) / 2;
  const offsetY = (rect.height - renderedHeight) / 2;
  return {
    x: offsetX + point.x * renderedWidth,
    y: offsetY + point.y * renderedHeight,
  };
}

function getHeadBounds(landmarks: FaceLandmarkerResult['faceLandmarks'][number], rect: DOMRect): HeadBounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const point of landmarks) {
    const mapped = mapLandmarkToStage(point, rect);
    minX = Math.min(minX, mapped.x);
    maxX = Math.max(maxX, mapped.x);
    minY = Math.min(minY, mapped.y);
    maxY = Math.max(maxY, mapped.y);
  }
  const width = maxX - minX;
  const height = maxY - minY;
  const leftTemple = landmarks[234] ? mapLandmarkToStage(landmarks[234], rect) : null;
  const rightTemple = landmarks[454] ? mapLandmarkToStage(landmarks[454], rect) : null;
  const angle = leftTemple && rightTemple
    ? Math.atan2(rightTemple.y - leftTemple.y, rightTemple.x - leftTemple.x)
    : 0;
  return {
    cx: (minX + maxX) / 2,
    cy: (minY + maxY) / 2,
    rx: Math.max(48, width * 0.62),
    ry: Math.max(60, height * 0.68),
    angle,
    scale: Math.max(width / Math.max(1, rect.width), height / Math.max(1, rect.height)),
  };
}

function getFaceArea(landmarks: FaceLandmarkerResult['faceLandmarks'][number]): number {
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
  return Math.max(0, maxX - minX) * Math.max(0, maxY - minY);
}

function getPrimaryFaceIndex(result: FaceLandmarkerResult): number {
  let primaryIndex = 0;
  let largestArea = -1;
  result.faceLandmarks.forEach((landmarks, index) => {
    const area = getFaceArea(landmarks);
    if (area > largestArea) {
      largestArea = area;
      primaryIndex = index;
    }
  });
  return primaryIndex;
}

function getMouthGeometry(landmarks: FaceLandmarkerResult['faceLandmarks'][number]): MouthGeometryMetrics {
  const leftCorner = landmarks[61];
  const rightCorner = landmarks[291];
  const upperLip = landmarks[13];
  const lowerLip = landmarks[14];
  if (!leftCorner || !rightCorner || !upperLip || !lowerLip) return { smile: 0, open: 0, width: 0 };

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
  const faceWidth = Math.max(0.001, maxX - minX);
  const faceHeight = Math.max(0.001, maxY - minY);
  const mouthWidth = Math.hypot(rightCorner.x - leftCorner.x, rightCorner.y - leftCorner.y) / faceWidth;
  const mouthCenterY = (upperLip.y + lowerLip.y) / 2;
  const cornerLift = (mouthCenterY - (leftCorner.y + rightCorner.y) / 2) / faceHeight;
  const mouthOpen = Math.abs(lowerLip.y - upperLip.y) / faceHeight;

  // A small, distant face often under-reports mouthSmile. This geometric
  // fallback looks for a wider mouth and lifted corners, but discounts open
  // mouth so plain yawning/张嘴 does not become rain.
  return {
    smile: clamp((mouthWidth - 0.31) * 1.9 + cornerLift * 5.4 - mouthOpen * 0.28, 0, 1),
    open: mouthOpen,
    width: mouthWidth,
  };
}

function readFeatures(result: FaceLandmarkerResult, inferenceMs: number, rect: DOMRect): FeatureSnapshot {
  const hasFace = result.faceLandmarks.length > 0;
  const faceIndex = hasFace ? getPrimaryFaceIndex(result) : 0;
  const landmarks = result.faceLandmarks[faceIndex] ?? null;
  const leftSmile = getBlendshapeScore(result, 'mouthSmileLeft', faceIndex);
  const rightSmile = getBlendshapeScore(result, 'mouthSmileRight', faceIndex);
  const smileScore = (leftSmile + rightSmile) / 2;
  const symmetricSmile = Math.min(leftSmile, rightSmile);
  const leftDimple = getBlendshapeScore(result, 'mouthDimpleLeft', faceIndex);
  const rightDimple = getBlendshapeScore(result, 'mouthDimpleRight', faceIndex);
  const mouthDimple = (leftDimple + rightDimple) / 2;
  const symmetricDimple = Math.min(leftDimple, rightDimple);
  const jawOpen = getBlendshapeScore(result, 'jawOpen', faceIndex);
  const cheekSquint = (
    getBlendshapeScore(result, 'cheekSquintLeft', faceIndex) +
    getBlendshapeScore(result, 'cheekSquintRight', faceIndex)
  ) / 2;
  const eyeSquint = (
    getBlendshapeScore(result, 'eyeSquintLeft', faceIndex) +
    getBlendshapeScore(result, 'eyeSquintRight', faceIndex)
  ) / 2;
  const mouthPress = (
    getBlendshapeScore(result, 'mouthPressLeft', faceIndex) +
    getBlendshapeScore(result, 'mouthPressRight', faceIndex)
  ) / 2;
  const mouthRoll = (
    getBlendshapeScore(result, 'mouthRollLower', faceIndex) +
    getBlendshapeScore(result, 'mouthRollUpper', faceIndex)
  ) / 2;
  const mouthPucker = getBlendshapeScore(result, 'mouthPucker', faceIndex);
  const lipPress = Math.max(mouthPress, mouthRoll, mouthPucker);
  const mouthGeometry = landmarks ? getMouthGeometry(landmarks) : { smile: 0, open: 0, width: 0 };
  const headBounds = landmarks ? getHeadBounds(landmarks, rect) : null;
  const isDistantFace = Boolean(headBounds && headBounds.scale < DISTANT_FACE_SCALE);

  // On a small face the dedicated mouthSmile blendshape often collapses
  // toward zero. mouthDimple survives at lower pixel density more reliably,
  // so use it only as distant-face support instead of lowering every global
  // smile threshold and reintroducing the lip-press false positive.
  // Weight the weaker side as well as the average. This preserves a real
  // bilateral smile but suppresses the one-sided false positive visible when
  // the user's head is tilted and just one mouth corner appears raised.
  const bilateralSmile = smileScore * 0.68 + symmetricSmile * 0.32;
  const bilateralDimple = mouthDimple * 0.68 + symmetricDimple * 0.32;
  const smileEvidence = isDistantFace
    ? Math.max(bilateralSmile, bilateralDimple * 0.82)
    : bilateralSmile;

  // Expressions in a live room can change quickly. Rise faster than we fall:
  // entering a stronger expression feels immediate, while the slower release
  // still filters one-frame spikes and keeps effects from flickering.
  smoothedSmile = smoothResponsive(smoothedSmile, smileEvidence, isDistantFace ? 0.72 : 0.62, 0.48);
  smoothedJaw = smoothResponsive(smoothedJaw, jawOpen, 0.52, 0.30);
  smoothedCheekSquint = smoothResponsive(smoothedCheekSquint, cheekSquint, 0.46, 0.28);
  smoothedEyeSquint = smoothResponsive(smoothedEyeSquint, eyeSquint, 0.46, 0.28);
  smoothedLipPress = smoothResponsive(smoothedLipPress, lipPress, 0.50, 0.30);
  return {
    faceDetected: hasFace,
    smileScore: smoothedSmile,
    jawOpen: smoothedJaw,
    lipPress: smoothedLipPress,
    geometrySmile: mouthGeometry.smile,
    geometryMouthOpen: mouthGeometry.open,
    mouthWidthRatio: mouthGeometry.width,
    inferenceMs,
    lastUpdatedAt: performance.now(),
    headBounds,
  };
}

function isLipPressing(now: number): boolean {
  // Pursed lips can raise MediaPipe's mouthSmile score, especially on a small
  // face. Cross-check the blendshape with the actual mouth geometry: a real
  // smile widens/lifts the mouth, while a press stays narrow and closed.
  const faceScale = latestFeatures.headBounds?.scale ?? 1;
  const isDistantFace = faceScale < DISTANT_FACE_SCALE;
  const pressEnter = isDistantFace ? LIP_PRESS_ENTER - 0.015 : LIP_PRESS_ENTER;
  const geometryConfirmsSmile = latestFeatures.geometrySmile >= (isDistantFace ? 0.050 : 0.075) &&
    latestFeatures.mouthWidthRatio >= (isDistantFace ? 0.355 : 0.385);
  const geometrySuggestsSmile = latestFeatures.geometrySmile >= (isDistantFace ? 0.025 : 0.035) &&
    latestFeatures.mouthWidthRatio >= (isDistantFace ? 0.335 : 0.36);
  const pressDominates = smoothedLipPress >= smoothedSmile * (isDistantFace ? 0.76 : 0.82);
  const closedNarrowMismatch = !geometrySuggestsSmile &&
    latestFeatures.geometrySmile < 0.045 &&
    latestFeatures.geometryMouthOpen < 0.035 &&
    latestFeatures.mouthWidthRatio < (isDistantFace ? 0.34 : 0.365) &&
    smoothedLipPress >= (isDistantFace ? 0.045 : 0.055) &&
    smoothedSmile < 0.44;
  // At close range the blendshape classifier is more reliable than the
  // corner-lift geometry fallback. A closed mouth with almost zero model
  // smile and a visible press must cancel rain even if perspective makes the
  // mouth corners look lifted. This is the exact failure visible in the
  // user's screenshot (smile 0.00, jaw 0.00, lip press 0.09).
  const directClosedPress = smoothedJaw <= 0.085 &&
    smoothedLipPress >= (isDistantFace ? 0.050 : CLOSED_LIP_PRESS_ENTER) &&
    smoothedSmile <= (isDistantFace ? 0.16 : 0.12) &&
    !geometryConfirmsSmile;
  // MediaPipe can report a pursed expression as both mouthSmile and
  // mouthPress. Treat it as a hard veto only when the geometry still looks
  // narrow; a widened closed-mouth smile must remain eligible for rain.
  const hardClosedPress = smoothedJaw <= 0.055 &&
    smoothedLipPress >= 0.095 &&
    !geometrySuggestsSmile &&
    (smoothedSmile <= 0.16 || pressDominates);
  const hasStrongEvidence = (
    smoothedJaw <= 0.15 &&
    !geometryConfirmsSmile &&
    ((smoothedLipPress >= pressEnter && pressDominates) || closedNarrowMismatch)
  ) || directClosedPress || hardClosedPress;
  if (hasStrongEvidence) lastStrongLipPressAt = now;
  return now - lastStrongLipPressAt <= LIP_PRESS_MEMORY_MS;
}

function hasLaughEvidence(now: number): boolean {
  // Use multiple evidence paths. Some cameras under-report jawOpen even for a
  // clear teeth-showing laugh, so the standard wide-mouth path must not be a
  // global gate for strong expressive or audio-supported laughter.
  const faceSupport = Math.max(smoothedCheekSquint, smoothedEyeSquint);
  const smileJawBalanced = smoothedSmile >= smoothedJaw * LAUGH_SMILE_TO_JAW_RATIO;
  const audioSupport = hasAudioLaughSupport(now);
  const standardVisualLaugh = smoothedSmile >= LAUGH_ENTER_SMILE &&
    smoothedJaw >= LAUGH_ENTER_JAW &&
    smileJawBalanced &&
    (faceSupport >= 0.02 || smoothedSmile >= 0.48);
  const broadOpenSmile = smoothedSmile >= 0.50 && smoothedJaw >= 0.18;
  const geometryVisualLaugh = latestFeatures.geometryMouthOpen >= GEOMETRY_LAUGH_OPEN_ENTER &&
    smoothedSmile >= 0.26 &&
    (latestFeatures.geometrySmile >= 0.045 || faceSupport >= 0.018);
  const expressiveVisualLaugh = smoothedSmile >= 0.66 &&
    (smoothedJaw >= 0.08 || latestFeatures.geometryMouthOpen >= 0.045) &&
    faceSupport >= 0.025;
  const audioAssistedLaugh = audioSupport &&
    smoothedSmile >= 0.46 &&
    (smoothedJaw >= 0.07 || latestFeatures.geometryMouthOpen >= 0.04);
  return standardVisualLaugh || broadOpenSmile || geometryVisualLaugh || expressiveVisualLaugh || audioAssistedLaugh;
}

function hasIntentionalSmileEvidence(entering: boolean): boolean {
  const isDistantFace = Boolean(
    latestFeatures.headBounds && latestFeatures.headBounds.scale < DISTANT_FACE_SCALE,
  );
  const smileThreshold = isDistantFace
    ? entering ? DISTANT_SMILE_ENTER : DISTANT_SMILE_EXIT
    : entering ? SMILE_ENTER : SMILE_EXIT;
  const strongThreshold = isDistantFace
    ? entering ? DISTANT_SMILE_STRONG_ENTER : DISTANT_SMILE_STRONG_EXIT
    : entering ? SMILE_STRONG_ENTER : SMILE_STRONG_EXIT;
  const geometryThreshold = isDistantFace
    ? entering ? DISTANT_GEOMETRY_SMILE_ENTER : DISTANT_GEOMETRY_SMILE_EXIT
    : entering ? GEOMETRY_SMILE_ENTER : GEOMETRY_SMILE_EXIT;
  const widthThreshold = isDistantFace
    ? entering ? DISTANT_SMILE_WIDTH_ENTER : DISTANT_SMILE_WIDTH_EXIT
    : entering ? SMILE_WIDTH_ENTER : SMILE_WIDTH_EXIT;

  // Even a high classifier value needs a weak independent geometry check.
  // Borderline values need the stricter three-way agreement. This removes the
  // last single-signal path that could turn a head tilt or smirk into rain.
  const strongModelSignal = smoothedSmile >= strongThreshold &&
    smoothedLipPress <= (isDistantFace ? 0.10 : 0.08) &&
    latestFeatures.geometrySmile >= geometryThreshold * 0.34 &&
    latestFeatures.mouthWidthRatio >= widthThreshold - 0.035;
  // A genuine closed-mouth smile can legitimately raise mouthPress a little
  // (the user's capture is smile 0.17 / press 0.12). Accept it when a softer
  // geometric shape check still shows lifted, widened corners; a real pout is
  // narrow and will fail this branch.
  const closedSmileSignal = smoothedSmile >= smileThreshold &&
    smoothedJaw <= 0.09 &&
    smoothedLipPress <= (isDistantFace ? 0.15 : 0.14) &&
    latestFeatures.geometrySmile >= geometryThreshold * 0.48 &&
    latestFeatures.mouthWidthRatio >= widthThreshold - 0.025;
  const corroboratedSignal = smoothedSmile >= smileThreshold &&
    latestFeatures.geometrySmile >= geometryThreshold &&
    latestFeatures.mouthWidthRatio >= widthThreshold;
  return strongModelSignal || closedSmileSignal || corroboratedSignal;
}

function desiredInteractionState(now: number): InteractionState {
  if (!latestFeatures.faceDetected) return 'IDLE';
  // A closed, compressed mouth must never leak into SMILE. MediaPipe often
  // raises mouthSmile at the same time as mouthPress for a pout, so this veto
  // deliberately runs before every positive expression branch.
  if (isLipPressing(now)) return 'IDLE';
  const laughDetected = hasLaughEvidence(now);
  // Laugh has priority over smile. This prevents the common transition where
  // a real laugh first gets stuck in SMILE and only upgrades much later.
  if (interactionState === 'LAUGH') {
    if (now - stateEnteredAt < LAUGH_MIN_DWELL_MS) return 'LAUGH';
    const faceSupport = Math.max(smoothedCheekSquint, smoothedEyeSquint);
    const audioSupport = hasAudioLaughSupport(now);
    const smileJawBalanced = smoothedSmile >= smoothedJaw * 0.72;
    const exitSupport = (faceSupport >= 0.03 && smileJawBalanced) || audioSupport;
    const strongMouthFallback = !audioEnabled && smoothedSmile >= 0.44 && smoothedJaw >= LAUGH_ENTER_JAW && smileJawBalanced;
    const expressiveExit = smoothedSmile >= 0.48 &&
      (smoothedJaw >= 0.08 || latestFeatures.geometryMouthOpen >= GEOMETRY_LAUGH_OPEN_EXIT) &&
      (faceSupport >= 0.025 || audioSupport);
    const standardExit = smoothedSmile >= LAUGH_EXIT_SMILE &&
      (smoothedJaw >= LAUGH_EXIT_JAW || latestFeatures.geometryMouthOpen >= GEOMETRY_LAUGH_OPEN_EXIT) &&
      (exitSupport || strongMouthFallback);
    if (standardExit || expressiveExit) {
      return 'LAUGH';
    }
  }
  if (laughDetected) return 'LAUGH';

  if (interactionState === 'SMILE') {
    if (now - stateEnteredAt < SMILE_MIN_DWELL_MS) return 'SMILE';
    if (hasIntentionalSmileEvidence(false) && smoothedJaw <= SMILE_JAW_EXIT_MAX) return 'SMILE';
  }

  if (interactionState === 'OPEN_MOUTH' && smoothedJaw >= OPEN_MOUTH_EXIT) return 'OPEN_MOUTH';
  if (hasIntentionalSmileEvidence(true) && smoothedJaw <= SMILE_JAW_ENTER_MAX) return 'SMILE';
  if (smoothedJaw >= OPEN_MOUTH_ENTER) return 'OPEN_MOUTH';
  return 'IDLE';
}

function stopRainImmediately(): void {
  for (let index = particles.length - 1; index >= 0; index -= 1) {
    if (particles[index].kind === 'rain' || particles[index].kind === 'splash') particles.splice(index, 1);
  }
  rainRipples.length = 0;
  rainAccumulator = 0;
  splashAccumulator = 0;
  smileEffectMix = 0;
  stageWrap.style.setProperty('--rain-grade-opacity', '0');
  stageWrap.style.setProperty('--rain-footage-opacity', '0');
  if (rainFootagePlaying) rainFootage.pause();
  rainFootagePlaying = false;
}

function transitionInteraction(next: InteractionState, now: number): void {
  if (next === interactionState) return;
  const previous = interactionState;
  interactionState = next;
  stateEnteredAt = now;
  stageWrap.classList.toggle('rain-active', next === 'SMILE');

  // Rain particles used to keep their full multi-second lifetime after a
  // short false smile. Fade existing drops quickly when leaving SMILE so a
  // corrected IDLE state is also visually immediate.
  if (previous === 'SMILE' && next !== 'SMILE') {
    stopRainImmediately();
  }
  updateHud();

  if (next === 'LAUGH' && now - lastFireworkAt >= FIREWORK_ENTRY_COOLDOWN_MS) {
    spawnProceduralFirework('entry');
    lastFireworkAt = now;
    fireworkSequenceUntil = now + FIREWORK_SEQUENCE_WINDOW_MS;
  }

  const messages: Record<InteractionState, string> = {
    IDLE: '状态稳定：保持自然表情，效果会逐渐停下。',
    SMILE: '检测到微笑：雨幕已启动。',
    OPEN_MOUTH: '检测到张嘴：保持中性，不触发焰火。',
    LAUGH: '检测到大笑：双层短尾焰火会绽放，火星撞到头部边界后会改变方向。',
  };
  hint.textContent = messages[next];
  console.info(`[interaction] ${previous} → ${next}`, {
    smile: smoothedSmile.toFixed(2),
    jaw: smoothedJaw.toFixed(2),
    lipPress: smoothedLipPress.toFixed(2),
  });
}

function updateInteraction(now: number): void {
  const desired = desiredInteractionState(now);
  if (interactionState === 'SMILE' && desired !== 'SMILE') stopRainImmediately();
  if (desired === 'LAUGH' && interactionState !== 'LAUGH' && now - lastFireworkAt >= FIREWORK_ENTRY_COOLDOWN_MS) {
    // The visual should feel live-room responsive: fire the burst as soon as
    // laugh evidence appears, while the state machine can still wait a few
    // frames before showing LAUGH in the HUD.
    laughEffectMix = Math.max(laughEffectMix, 0.72);
    spawnProceduralFirework('entry');
    lastFireworkAt = now;
    fireworkSequenceUntil = now + FIREWORK_SEQUENCE_WINDOW_MS;
  }
  if (desired !== candidateState) {
    candidateState = desired;
    candidateSince = now;
  }
  const holdMs = candidateState === 'LAUGH'
    ? LAUGH_HOLD_MS
    : candidateState === 'SMILE'
      ? latestFeatures.headBounds && latestFeatures.headBounds.scale < DISTANT_FACE_SCALE
        ? DISTANT_SMILE_HOLD_MS
        : SMILE_HOLD_MS
      : STATE_HOLD_MS;
  if (candidateState !== interactionState && now - candidateSince >= holdMs) {
    transitionInteraction(candidateState, now);
  }
}

function drawDebugFace(): void {
  if (!SHOW_DEBUG_LANDMARKS || !ctx) return;
  const rect = video.getBoundingClientRect();
  ctx.save();
  ctx.fillStyle = 'rgba(141, 255, 207, 0.72)';
  if (latestLandmarks) {
    for (const point of latestLandmarks) {
      // Canvas is mirrored with the video via CSS. The cover transform still
      // needs to account for the camera crop before plotting source landmarks.
      const { x, y } = mapLandmarkToStage(point, rect);
      ctx.beginPath();
      ctx.arc(x, y, 1.2, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  if (latestFeatures.headBounds) {
    const bounds = latestFeatures.headBounds;
    ctx.strokeStyle = 'rgba(255, 194, 111, 0.86)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([7, 6]);
    ctx.beginPath();
    ctx.ellipse(bounds.cx, bounds.cy, bounds.rx, bounds.ry, bounds.angle, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(255, 214, 158, 0.92)';
    ctx.font = '700 11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText('COLLISION BOUNDARY', bounds.cx - bounds.rx, bounds.cy - bounds.ry - 9);
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
    vx: -24 + Math.random() * 34,
    vy: 360 + Math.random() * 260,
    size: 16 + Math.random() * 20,
    life: 1.65 + Math.random() * 0.55,
    maxLife: 2.2,
    hue: 205,
    alpha: 0.16 + Math.random() * 0.17,
  });
}

function spawnRainSplash(width: number, height: number): void {
  const x = width * (0.08 + Math.random() * 0.84);
  const y = height * (0.80 + Math.random() * 0.15);
  const count = 3 + Math.floor(Math.random() * 4);

  rainRipples.push({
    x,
    y,
    radius: 2,
    maxRadius: 12 + Math.random() * 16,
    life: 0.34 + Math.random() * 0.12,
    maxLife: 0.46,
    alpha: 0.14 + Math.random() * 0.10,
  });
  if (rainRipples.length > MAX_RAIN_RIPPLES) rainRipples.shift();

  for (let index = 0; index < count; index += 1) {
    pushParticle({
      kind: 'splash',
      x: x + (Math.random() - 0.5) * 12,
      y: y + Math.random() * 4,
      vx: (Math.random() - 0.5) * (90 + Math.random() * 130),
      vy: -(95 + Math.random() * 155),
      size: 1.05 + Math.random() * 1.6,
      life: 0.30 + Math.random() * 0.24,
      maxLife: 0.54,
      hue: 202,
      alpha: 0.22 + Math.random() * 0.20,
      prevX: x,
      prevY: y,
    });
  }
}

function pushFireworkBurst(burst: FireworkBurst): void {
  if (fireworkBursts.length >= MAX_FIREWORK_BURSTS) fireworkBursts.shift();
  fireworkBursts.push(burst);
}

function headLocalToStage(bounds: HeadBounds, localX: number, localY: number): { x: number; y: number } {
  const cosine = Math.cos(bounds.angle);
  const sine = Math.sin(bounds.angle);
  return {
    x: bounds.cx + localX * cosine - localY * sine,
    y: bounds.cy + localX * sine + localY * cosine,
  };
}

function stageToHeadLocal(bounds: HeadBounds, x: number, y: number): { x: number; y: number } {
  const cosine = Math.cos(bounds.angle);
  const sine = Math.sin(bounds.angle);
  const dx = x - bounds.cx;
  const dy = y - bounds.cy;
  return {
    x: dx * cosine + dy * sine,
    y: -dx * sine + dy * cosine,
  };
}

function spawnFirework(mode: 'entry' | 'sustain'): void {
  const animation = authoredFirework;
  if (!animation) {
    void authoredFireworkReady.then((loaded) => {
      if (loaded && interactionState === 'LAUGH' && fireworkBursts.length === 0) spawnFirework(mode);
    });
    return;
  }
  const rect = video.getBoundingClientRect();
  const bounds = latestFeatures.headBounds;
  const isEntry = mode === 'entry';
  const coreCanvas = document.createElement('canvas');
  coreCanvas.width = AUTHORED_FIREWORK_CACHE_SIZE;
  coreCanvas.height = AUTHORED_FIREWORK_CACHE_SIZE;
  const size = bounds
    ? clamp(bounds.rx * (isEntry ? 2.78 : 2.58), 210, Math.min(390, rect.width * 0.72))
    : Math.min(320, rect.width * 0.62);
  const side = fireworkSequence % 2 === 0 ? -1 : 1;
  fireworkSequence += 1;
  const sideOffset = bounds ? Math.max(bounds.rx * 0.42, size * 0.15) : 0;
  const burstX = bounds
    ? clamp(bounds.cx + side * sideOffset, size * 0.30, rect.width - size * 0.30)
    : rect.width * 0.45;
  const burstY = bounds
    ? clamp(bounds.cy - bounds.ry * 1.28, size * 0.27, rect.height * 0.36)
    : rect.height * 0.22;
  pushFireworkBurst({
    x: burstX,
    y: burstY,
    size,
    rotation: bounds ? bounds.angle * 0.14 : 0,
    life: AUTHORED_FIREWORK_DURATION,
    maxLife: AUTHORED_FIREWORK_DURATION,
    alpha: isEntry ? 1 : 0.94,
    coreCanvas,
    lastCoreFrame: -1,
    physicalParticles: animation.layers
      .filter((layer) => layer.physical)
      .flatMap((layer) => layer.groups.map((group) => ({
        layer,
        group,
        previousAuthoredX: null,
        previousAuthoredY: null,
        x: 0,
        y: 0,
        vx: 0,
        vy: 0,
        rotation: 0,
        collided: false,
      }))),
  });
}

function spawnProceduralFirework(mode: 'entry' | 'sustain'): void {
  const rect = video.getBoundingClientRect();
  proceduralFireworks.spawn(
    latestFeatures.headBounds,
    { width: rect.width, height: rect.height },
    mode,
  );
}

function authoredSourceFrame(burst: FireworkBurst, animation: AuthoredFirework): number {
  const progress = clamp(1 - burst.life / burst.maxLife, 0, 1);
  return animation.inFrame + progress * (animation.outFrame - animation.inFrame - 0.001);
}

function authoredBurstAlpha(burst: FireworkBurst): number {
  const progress = clamp(1 - burst.life / burst.maxLife, 0, 1);
  const fadeIn = clamp(progress / 0.07, 0, 1);
  const fadeOut = clamp((1 - progress) / 0.18, 0, 1);
  return burst.alpha * Math.min(fadeIn, fadeOut) * Math.max(0.78, laughEffectMix);
}

function compositionToStage(
  burst: FireworkBurst,
  animation: AuthoredFirework,
  x: number,
  y: number,
): { x: number; y: number } {
  const scale = burst.size / animation.width;
  const localX = (x - animation.width / 2) * scale;
  const localY = (y - animation.height / 2) * scale;
  const cosine = Math.cos(burst.rotation);
  const sine = Math.sin(burst.rotation);
  return {
    x: burst.x + localX * cosine - localY * sine,
    y: burst.y + localX * sine + localY * cosine,
  };
}

interface HeadCollision {
  x: number;
  y: number;
  nx: number;
  ny: number;
}

function findHeadCollision(
  previousX: number,
  previousY: number,
  currentX: number,
  currentY: number,
  bounds: HeadBounds,
): HeadCollision | null {
  const previousLocal = stageToHeadLocal(bounds, previousX, previousY);
  const currentLocal = stageToHeadLocal(bounds, currentX, currentY);
  const ellipseValue = (point: { x: number; y: number }): number =>
    (point.x * point.x) / (bounds.rx * bounds.rx) + (point.y * point.y) / (bounds.ry * bounds.ry);
  const previousValue = ellipseValue(previousLocal);
  const currentValue = ellipseValue(currentLocal);
  if (currentValue >= 1) return null;

  let contactLocal: { x: number; y: number };
  if (previousValue >= 1) {
    // Particle crossed the head boundary between two render frames. Find the
    // first contact point instead of allowing high-speed particles to tunnel.
    let outsideT = 0;
    let insideT = 1;
    for (let iteration = 0; iteration < 6; iteration += 1) {
      const midpoint = (outsideT + insideT) / 2;
      const point = {
        x: previousLocal.x + (currentLocal.x - previousLocal.x) * midpoint,
        y: previousLocal.y + (currentLocal.y - previousLocal.y) * midpoint,
      };
      if (ellipseValue(point) >= 1) outsideT = midpoint;
      else insideT = midpoint;
    }
    contactLocal = {
      x: previousLocal.x + (currentLocal.x - previousLocal.x) * insideT,
      y: previousLocal.y + (currentLocal.y - previousLocal.y) * insideT,
    };
  } else {
    // If the user moves their head into an existing particle, both particle
    // samples can already be inside the new ellipse. Project it back to the
    // boundary so head motion also produces a visible physical response.
    const projectionScale = 1 / Math.sqrt(Math.max(0.0001, currentValue));
    contactLocal = {
      x: currentLocal.x * projectionScale,
      y: currentLocal.y * projectionScale,
    };
  }
  const contact = headLocalToStage(bounds, contactLocal.x, contactLocal.y);
  const gradientX = contactLocal.x / (bounds.rx * bounds.rx);
  const gradientY = contactLocal.y / (bounds.ry * bounds.ry);
  const gradientLength = Math.max(0.0001, Math.hypot(gradientX, gradientY));
  const localNx = gradientX / gradientLength;
  const localNy = gradientY / gradientLength;
  const cosine = Math.cos(bounds.angle);
  const sine = Math.sin(bounds.angle);
  return {
    x: contact.x,
    y: contact.y,
    nx: localNx * cosine - localNy * sine,
    ny: localNx * sine + localNy * cosine,
  };
}

function updateAuthoredParticles(
  burst: FireworkBurst,
  animation: AuthoredFirework,
  frame: number,
  deltaSeconds: number,
  bounds: HeadBounds | null,
): void {
  for (const particle of burst.physicalParticles) {
    const visibleOnAuthoredTimeline = frame >= particle.layer.inFrame && frame <= particle.layer.outFrame;
    if (!visibleOnAuthoredTimeline && !particle.collided) {
      particle.previousAuthoredX = null;
      particle.previousAuthoredY = null;
      continue;
    }

    if (particle.collided) {
      particle.vy += 245 * deltaSeconds;
      particle.vx *= 1 - Math.min(0.24, deltaSeconds * 1.45);
      particle.x += particle.vx * deltaSeconds;
      particle.y += particle.vy * deltaSeconds;
      particle.rotation = Math.atan2(particle.vy, particle.vx);
      continue;
    }

    const compositionPoint = groupCompositionPosition(particle.layer, particle.group, frame);
    const authoredPoint = compositionToStage(burst, animation, compositionPoint.x, compositionPoint.y);
    particle.x = authoredPoint.x;
    particle.y = authoredPoint.y;

    if (particle.previousAuthoredX !== null && particle.previousAuthoredY !== null && bounds) {
      const safeDelta = Math.max(1 / 120, deltaSeconds);
      const velocityX = (authoredPoint.x - particle.previousAuthoredX) / safeDelta;
      const velocityY = (authoredPoint.y - particle.previousAuthoredY) / safeDelta;
      const collision = findHeadCollision(
        particle.previousAuthoredX,
        particle.previousAuthoredY,
        authoredPoint.x,
        authoredPoint.y,
        bounds,
      );
      if (collision) {
        const normalVelocity = velocityX * collision.nx + velocityY * collision.ny;
        const tangentVelocityX = velocityX - normalVelocity * collision.nx;
        const tangentVelocityY = velocityY - normalVelocity * collision.ny;
        const reboundNormalSpeed = Math.max(82, Math.max(0, -normalVelocity) * 0.66);
        particle.x = collision.x + collision.nx * 2.5;
        particle.y = collision.y + collision.ny * 2.5;
        particle.vx = tangentVelocityX * 0.76 + collision.nx * reboundNormalSpeed;
        particle.vy = tangentVelocityY * 0.76 + collision.ny * reboundNormalSpeed;
        particle.rotation = Math.atan2(particle.vy, particle.vx);
        particle.collided = true;
        if (PREVIEW_FIREWORK) {
          previewCollisionHits += 1;
          canvas.dataset.collisionHits = String(previewCollisionHits);
        }
        pushCollisionFlash(
          particle.x,
          particle.y,
          particle.layer.name.includes('White') ? 292 : 48,
          collision.nx,
          collision.ny,
        );
      } else if (Math.hypot(velocityX, velocityY) > 4) {
        particle.rotation = Math.atan2(velocityY, velocityX);
      }
    }
    particle.previousAuthoredX = authoredPoint.x;
    particle.previousAuthoredY = authoredPoint.y;
  }
}

function pushCollisionFlash(x: number, y: number, hue: number, nx = 0, ny = -1): void {
  collisionFlashes.push({
    x,
    y,
    radius: 3,
    life: 0.34,
    maxLife: 0.34,
    hue,
  });
  if (collisionFlashes.length > MAX_COLLISION_FLASHES) collisionFlashes.shift();

  for (let index = 0; index < 3; index += 1) {
    const tangent = (Math.random() - 0.5) * 120;
    const rebound = 95 + Math.random() * 155;
    pushParticle({
      kind: 'ember',
      x,
      y,
      vx: nx * rebound - ny * tangent,
      vy: ny * rebound + nx * tangent,
      size: 1.5 + Math.random() * 1.3,
      life: 0.32 + Math.random() * 0.18,
      maxLife: 0.50,
      hue: hue + (Math.random() - 0.5) * 18,
      alpha: 0.62 + Math.random() * 0.30,
      prevX: x,
      prevY: y,
    });
  }
}

function updateSustainedFireworks(now: number): void {
  if (interactionState !== 'LAUGH') return;
  const repeatInterval = now <= fireworkSequenceUntil
    ? FIREWORK_SEQUENCE_INTERVAL_MS
    : FIREWORK_REPEAT_MS;
  if (now - lastFireworkAt < repeatInterval) return;
  // The particle system owns a fixed-size ring buffer, so a light follow-up
  // burst can overlap the previous tails without increasing the frame budget.
  spawnProceduralFirework('sustain');
  lastFireworkAt = now;
}

function updateParticles(deltaSeconds: number, rect: DOMRect): void {
  if (interactionState === 'SMILE') {
    // Footage carries the natural mid-ground; Canvas adds readable foreground
    // streaks and bottom splashes without tinting the camera feed.
    rainAccumulator += deltaSeconds * RAIN_DROP_RATE;
    while (rainAccumulator >= 1) {
      spawnRainDrop(rect.width);
      rainAccumulator -= 1;
    }
    splashAccumulator += deltaSeconds * RAIN_SPLASH_RATE;
    while (splashAccumulator >= 1) {
      spawnRainSplash(rect.width, rect.height);
      splashAccumulator -= 1;
    }
  } else {
    rainAccumulator = Math.min(rainAccumulator, 1);
    splashAccumulator = Math.min(splashAccumulator, 1);
  }

  for (let index = particles.length - 1; index >= 0; index -= 1) {
    const particle = particles[index];
    particle.life -= deltaSeconds;
    if (particle.life <= 0 || particle.y > rect.height + 40 || particle.x < -80 || particle.x > rect.width + 80) {
      particles.splice(index, 1);
      continue;
    }

    particle.prevX = particle.x;
    particle.prevY = particle.y;
    if (particle.kind === 'rain') {
      particle.vy += 95 * deltaSeconds;
    } else if (particle.kind === 'splash') {
      particle.vy += 420 * deltaSeconds;
      particle.vx *= 1 - Math.min(0.45, deltaSeconds * 2.8);
    } else if (particle.kind === 'ember') {
      particle.vy += 230 * deltaSeconds;
      particle.vx *= 1 - Math.min(0.30, deltaSeconds * 1.8);
    }
    particle.x += particle.vx * deltaSeconds;
    particle.y += particle.vy * deltaSeconds;
  }
}

function updateRainRipples(deltaSeconds: number): void {
  for (let index = rainRipples.length - 1; index >= 0; index -= 1) {
    const ripple = rainRipples[index];
    ripple.life -= deltaSeconds;
    if (ripple.life <= 0) {
      rainRipples.splice(index, 1);
      continue;
    }
    const progress = 1 - ripple.life / ripple.maxLife;
    ripple.radius = 2 + (ripple.maxRadius - 2) * progress;
  }
}

function updateEffectMix(deltaSeconds: number): void {
  const response = Math.min(1, deltaSeconds * 8);
  const smileTarget = interactionState === 'SMILE' ? 1 : 0;
  const laughTarget = interactionState === 'LAUGH' ? 1 : 0;
  smileEffectMix = smileTarget > 0 ? smileEffectMix + (smileTarget - smileEffectMix) * response : 0;
  laughEffectMix += (laughTarget - laughEffectMix) * response;

  // The stock footage provides a consistent mid-ground layer; code particles
  // remain in front for depth and responsiveness. Opacity follows the same
  // eased mix, so a short classification spike cannot hard-flash the video.
  stageWrap.style.setProperty('--rain-grade-opacity', (smileEffectMix * 0.06).toFixed(3));
  stageWrap.style.setProperty('--rain-footage-opacity', (smileEffectMix * 0.42).toFixed(3));
  const shouldPlay = uiState === 'running' && interactionState === 'SMILE';
  if (shouldPlay && !rainFootagePlaying) {
    rainFootagePlaying = true;
    void rainFootage.play().catch(() => {
      rainFootagePlaying = false;
    });
  } else if (!shouldPlay && rainFootagePlaying) {
    rainFootage.pause();
    rainFootagePlaying = false;
  }
}

function updateCollisionFlashes(deltaSeconds: number): void {
  for (let index = collisionFlashes.length - 1; index >= 0; index -= 1) {
    const flash = collisionFlashes[index];
    flash.life -= deltaSeconds;
    if (flash.life <= 0) {
      collisionFlashes.splice(index, 1);
      continue;
    }
    const progress = 1 - flash.life / flash.maxLife;
    flash.radius = 3 + progress * 17;
  }
}

function updateFireworkBursts(deltaSeconds: number): void {
  for (let index = fireworkBursts.length - 1; index >= 0; index -= 1) {
    const burst = fireworkBursts[index];
    burst.life -= deltaSeconds;
    if (burst.life <= 0) {
      fireworkBursts.splice(index, 1);
      continue;
    }
    if (authoredFirework) {
      updateAuthoredParticles(
        burst,
        authoredFirework,
        authoredSourceFrame(burst, authoredFirework),
        deltaSeconds,
        latestFeatures.headBounds,
      );
    }
  }
}

function drawStageAtmosphere(rect: DOMRect): void {
  if (!ctx) return;

  if (smileEffectMix > 0.01) {
    const rainGlow = ctx.createLinearGradient(0, 0, 0, rect.height);
    rainGlow.addColorStop(0, `rgba(50, 196, 255, ${0.16 * smileEffectMix})`);
    rainGlow.addColorStop(0.52, `rgba(31, 107, 207, ${0.04 * smileEffectMix})`);
    rainGlow.addColorStop(1, 'rgba(4, 16, 45, 0)');
    ctx.save();
    ctx.globalCompositeOperation = 'screen';
    ctx.fillStyle = rainGlow;
    ctx.fillRect(0, 0, rect.width, rect.height);
    ctx.restore();
  }

  // Firework sprites carry their own local bloom. Avoid a full-frame warm
  // grade here so the camera skin tone stays natural during LAUGH.
}

function drawCollisionFlashes(): void {
  if (!ctx) return;
  ctx.save();
  ctx.globalCompositeOperation = 'screen';
  for (const flash of collisionFlashes) {
    const lifeRatio = Math.max(0, flash.life / flash.maxLife);
    const progress = 1 - lifeRatio;
    ctx.globalAlpha = Math.min(1, lifeRatio * 1.8) * 0.88;
    ctx.fillStyle = `hsla(${flash.hue}, 100%, 86%, 0.92)`;
    ctx.shadowColor = `hsla(${flash.hue}, 100%, 78%, 0.8)`;
    ctx.shadowBlur = 8;
    for (let index = 0; index < 6; index += 1) {
      const angle = (Math.PI * 2 * index) / 6 + progress * 0.75;
      const stableJitter = pseudoRandom(index * 8.521 + flash.x * 0.019 + flash.y * 0.013);
      traceTaperedPetal(ctx, flash.x, angle, flash.radius * 0.12, flash.radius * (0.48 + stableJitter * 0.30), 2.3 + lifeRatio * 1.5);
    }
  }
  ctx.restore();
}

function drawRainRipples(): void {
  if (!ctx || smileEffectMix <= 0.01) return;
  ctx.save();
  ctx.globalCompositeOperation = 'screen';
  ctx.lineCap = 'round';
  for (const ripple of rainRipples) {
    const lifeRatio = Math.max(0, ripple.life / ripple.maxLife);
    ctx.globalAlpha = ripple.alpha * Math.min(1, lifeRatio * 1.6) * smileEffectMix;
    ctx.strokeStyle = 'rgba(230, 240, 246, 1)';
    ctx.lineWidth = 0.75 + lifeRatio * 0.5;
    ctx.beginPath();
    ctx.ellipse(ripple.x, ripple.y, ripple.radius, ripple.radius * 0.18, 0, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

function traceTaperedPetal(
  target: CanvasRenderingContext2D,
  center: number,
  angle: number,
  startRadius: number,
  length: number,
  width: number,
  bend = 0,
): void {
  target.save();
  target.translate(center, center);
  target.rotate(angle);
  const end = startRadius + length;
  target.beginPath();
  target.moveTo(startRadius, 0);
  target.bezierCurveTo(
    startRadius + length * 0.18,
    -width * 0.28 + bend * 0.18,
    startRadius + length * 0.64,
    -width * 0.54 + bend * 0.72,
    end - width * 0.18,
    -width * 0.28 + bend,
  );
  target.quadraticCurveTo(end + width * 0.10, bend, end - width * 0.18, width * 0.28 + bend);
  target.bezierCurveTo(
    startRadius + length * 0.64,
    width * 0.54 + bend * 0.72,
    startRadius + length * 0.20,
    width * 0.28 + bend * 0.18,
    startRadius,
    0,
  );
  target.closePath();
  target.fill();
  target.restore();
}

function drawFireworkBursts(): void {
  const animation = authoredFirework;
  if (!ctx || !animation || laughEffectMix <= 0.01) return;
  for (const burst of fireworkBursts) {
    const frame = authoredSourceFrame(burst, animation);
    const frameIndex = Math.floor(frame);
    const alpha = authoredBurstAlpha(burst);
    if (alpha <= 0.01) continue;

    // Decorative stars are cached at the source animation's 30 fps, then
    // composited as one bitmap. Only the authored moving groups stay as
    // individual vectors because they are the particles that can collide.
    if (burst.lastCoreFrame !== frameIndex) {
      renderAuthoredCore(animation, burst.coreCanvas, frame);
      burst.lastCoreFrame = frameIndex;
    }
    ctx.save();
    ctx.globalCompositeOperation = 'screen';
    ctx.globalAlpha = alpha;
    ctx.translate(burst.x, burst.y);
    ctx.rotate(burst.rotation);
    ctx.shadowColor = 'rgba(255, 205, 160, 0.42)';
    ctx.shadowBlur = 9;
    ctx.drawImage(burst.coreCanvas, -burst.size / 2, -burst.size / 2, burst.size, burst.size);
    ctx.restore();

    const authoredScale = burst.size / animation.width;
    ctx.save();
    ctx.globalCompositeOperation = 'screen';
    for (const particle of burst.physicalParticles) {
      drawAuthoredPhysicalGroup(
        ctx,
        particle.layer,
        particle.group,
        particle.collided ? Math.min(frame, particle.layer.outFrame - 0.001) : frame,
        particle.x,
        particle.y,
        authoredScale,
        particle.collided ? particle.rotation : burst.rotation,
        alpha,
      );
    }
    ctx.restore();
  }
}

function drawSparkleParticle(particle: Particle, lifeRatio: number, effectMix: number): void {
  if (!ctx) return;
  const alpha = particle.alpha * Math.min(1, lifeRatio * 2.2) * effectMix;
  const radius = particle.size;
  ctx.globalAlpha = 1;
  ctx.fillStyle = `hsla(${particle.hue}, 100%, 78%, ${0.24 * alpha})`;
  ctx.beginPath();
  ctx.arc(particle.x, particle.y, radius * 2.4, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = `hsla(${particle.hue}, 100%, 92%, ${0.88 * alpha})`;
  ctx.beginPath();
  ctx.arc(particle.x, particle.y, radius, 0, Math.PI * 2);
  ctx.fill();
}

function drawParticles(): void {
  if (!ctx) return;
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  ctx.lineCap = 'round';

  for (const particle of particles) {
    const lifeRatio = Math.max(0, particle.life / particle.maxLife);
    ctx.globalAlpha = particle.alpha * Math.min(1, lifeRatio * 1.8);
    ctx.strokeStyle = `hsla(${particle.hue}, 100%, 78%, 1)`;
    ctx.fillStyle = `hsla(${particle.hue}, 100%, 70%, 1)`;
    if (particle.kind === 'rain') {
      ctx.globalAlpha = particle.alpha * Math.min(1, lifeRatio * 1.8) * smileEffectMix;
      ctx.strokeStyle = 'rgba(228, 239, 245, 1)';
      ctx.lineWidth = 1.18 + particle.size * 0.068;
      ctx.beginPath();
      ctx.moveTo(particle.x, particle.y);
      ctx.lineTo(particle.x - particle.vx * 0.026, particle.y - particle.vy * 0.056);
      ctx.stroke();
    } else if (particle.kind === 'splash') {
      ctx.globalAlpha = particle.alpha * Math.min(1, lifeRatio * 2.2) * smileEffectMix;
      ctx.strokeStyle = 'rgba(232, 244, 249, 1)';
      ctx.fillStyle = 'rgba(232, 244, 249, 1)';
      ctx.lineWidth = Math.max(0.8, particle.size * 0.64);
      ctx.beginPath();
      ctx.moveTo(particle.prevX ?? particle.x, particle.prevY ?? particle.y);
      ctx.lineTo(particle.x, particle.y);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(particle.x, particle.y, particle.size * (0.75 + lifeRatio * 0.18), 0, Math.PI * 2);
      ctx.fill();
    } else if (particle.kind === 'ember') {
      drawSparkleParticle(particle, lifeRatio, laughEffectMix);
    }
  }
  ctx.restore();
}

function render(now: number): void {
  if (now - lastVisualFrameAt < RENDER_INTERVAL_MS - 1) {
    requestAnimationFrame(render);
    return;
  }
  lastVisualFrameAt = now;
  frameCount += 1;
  if (now - lastFpsAt >= 1000) {
    renderFps = frameCount;
    frameCount = 0;
    lastFpsAt = now;
    if (PREVIEW_FIREWORK) updateHud();
  }

  const rect = video.getBoundingClientRect();
  if (PREVIEW_FIREWORK && !previewInitialized && rect.width > 0) {
    previewInitialized = true;
    stageMessage.hidden = true;
    stageMessage.classList.add('hidden');
    uiState = 'running';
    interactionState = 'LAUGH';
    stateEnteredAt = now;
    laughEffectMix = 1;
    latestFeatures = {
      ...latestFeatures,
      faceDetected: true,
      headBounds: {
        cx: rect.width * 0.5,
        cy: rect.height * 0.62,
        rx: rect.width * 0.15,
        ry: rect.height * 0.25,
        angle: -0.04,
        scale: 0.34,
      },
    };
    spawnProceduralFirework('entry');
    lastFireworkAt = now;
    fireworkSequenceUntil = now + FIREWORK_SEQUENCE_WINDOW_MS;
  }
  if (canvas.width !== Math.floor(rect.width * Math.min(window.devicePixelRatio || 1, MAX_CANVAS_DPR))) resizeCanvas();
  ctx?.clearRect(0, 0, rect.width, rect.height);
  sampleAudio(now);

  if (faceLandmarker && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && now - lastInferenceAt >= INFERENCE_INTERVAL_MS) {
    const inferenceStart = performance.now();
    const result = faceLandmarker.detectForVideo(video, now);
    const inferenceCompletedAt = performance.now();
    lastInferenceAt = inferenceCompletedAt;
    const primaryFaceIndex = result.faceLandmarks.length > 0 ? getPrimaryFaceIndex(result) : 0;
    latestLandmarks = result.faceLandmarks[primaryFaceIndex] ?? null;
    latestFeatures = readFeatures(result, inferenceCompletedAt - inferenceStart, rect);
    updateHud();
  }

  // State confirmation is time-based, so check it on render frames instead
  // of waiting for the next 10 Hz inference tick. This removes a perceptible
  // extra delay when SMILE upgrades to LAUGH without increasing model load.
  if (uiState === 'running' && !PREVIEW_FIREWORK) updateInteraction(performance.now());

  const deltaSeconds = Math.min(0.05, (now - previousRenderAt) / 1000);
  previousRenderAt = now;
  updateSustainedFireworks(now);
  updateEffectMix(deltaSeconds);
  updateParticles(deltaSeconds, rect);
  updateRainRipples(deltaSeconds);
  proceduralFireworks.update(deltaSeconds, latestFeatures.headBounds);
  if (PREVIEW_FIREWORK) {
    canvas.dataset.collisionHits = String(proceduralFireworks.collisionCount);
    canvas.dataset.activeParticles = String(proceduralFireworks.activeCount);
  }
  drawStageAtmosphere(rect);
  drawRainRipples();
  if (ctx) proceduralFireworks.draw(ctx);
  drawParticles();
  drawDebugFace();
  requestAnimationFrame(render);
}

async function createLandmarker(): Promise<FaceLandmarker> {
  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
  return FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numFaces: MAX_TRACKED_FACES,
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
    audioButton.disabled = false;
    updateAudioStatus('声音辅助关闭 · 不读取麦克风', false);
    setUiState('running', '请自然微笑、露齿大笑或张嘴；需要时可开启声音辅助。');
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
  disableAudioAssist();
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  faceLandmarker?.close();
  faceLandmarker = null;
  latestLandmarks = null;
  particles.length = 0;
  collisionFlashes.length = 0;
  rainRipples.length = 0;
  fireworkBursts.length = 0;
  proceduralFireworks.clear();
  smileEffectMix = 0;
  laughEffectMix = 0;
  rainFootage.pause();
  rainFootage.currentTime = 0;
  rainAccumulator = 0;
  splashAccumulator = 0;
  stageWrap.style.setProperty('--rain-grade-opacity', '0');
  stageWrap.style.setProperty('--rain-footage-opacity', '0');
  stageWrap.style.setProperty('--rain-lens-opacity', '0');
  stageWrap.classList.remove('rain-active');
  rainFootagePlaying = false;
  smoothedSmile = 0;
  smoothedJaw = 0;
  smoothedCheekSquint = 0;
  smoothedEyeSquint = 0;
  smoothedLipPress = 0;
  lastStrongLipPressAt = -Infinity;
  interactionState = 'IDLE';
  candidateState = 'IDLE';
  candidateSince = 0;
  stateEnteredAt = 0;
  lastFireworkAt = -Infinity;
  fireworkSequenceUntil = -Infinity;
  previousRenderAt = performance.now();
  lastVisualFrameAt = -Infinity;
  latestFeatures = {
    faceDetected: false,
    smileScore: 0,
    jawOpen: 0,
    lipPress: 0,
    geometrySmile: 0,
    geometryMouthOpen: 0,
    mouthWidthRatio: 0,
    inferenceMs: 0,
    lastUpdatedAt: 0,
    headBounds: null,
  };
  updateHud();
  stageMessage.classList.remove('hidden');
  stageMessage.querySelector('strong')!.textContent = '点击开始体验';
  stageMessage.querySelector('span')!.textContent = '默认只读取摄像头；声音辅助需主动开启，所有分析都在浏览器本地完成。';
  resetButton.disabled = true;
  startButton.disabled = false;
  audioButton.disabled = true;
  setUiState('idle', '技术验证目标：确认表情状态可以稳定驱动不同效果。');
}

startButton.addEventListener('click', () => void startExperience());
resetButton.addEventListener('click', resetExperience);
audioButton.addEventListener('click', () => void enableAudioAssist());
window.addEventListener('resize', resizeCanvas);
setUiState('idle');
requestAnimationFrame(render);
