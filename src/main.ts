import './style.css';
import { FaceLandmarker, FilesetResolver, type FaceLandmarkerResult } from '@mediapipe/tasks-vision';

const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const WASM_ROOT = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
const INFERENCE_INTERVAL_MS = 1000 / 10;
const MAX_CANVAS_DPR = 1.5;
const SHOW_DEBUG_LANDMARKS = new URLSearchParams(window.location.search).has('debug');

// The model reports a new value every frame, so the thresholds deliberately
// use a gap between entering and leaving a state (hysteresis). This keeps a
// borderline expression from rapidly toggling the visual effect.
// These values are tuned against the actual test captures rather than the
// model's ideal demo range. In backlight and behind glasses a real subtle
// smile can be reported around 0.10, while a neutral face stays near 0.00.
const SMILE_ENTER = 0.05;
const SMILE_EXIT = 0.025;
const SMILE_JAW_ENTER_MAX = 0.38;
const SMILE_JAW_EXIT_MAX = 0.48;
const LAUGH_ENTER_SMILE = 0.42;
const LAUGH_ENTER_JAW = 0.40;
const LAUGH_EXIT_SMILE = 0.30;
const LAUGH_EXIT_JAW = 0.26;
const LAUGH_SMILE_TO_JAW_RATIO = 0.82;
const LIP_PRESS_ENTER = 0.14;
const LIP_PRESS_MEMORY_MS = 650;
const STATE_HOLD_MS = 85;
const SMILE_HOLD_MS = 220;
const LAUGH_HOLD_MS = 100;
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
const FIREWORK_COOLDOWN_MS = 900;
const MAX_PARTICLES = 360;

type UiState = 'idle' | 'loading' | 'running' | 'error';
type InteractionState = 'IDLE' | 'SMILE' | 'OPEN_MOUTH' | 'LAUGH';
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
  lipPress: number;
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

interface EffectRing {
  x: number;
  y: number;
  radius: number;
  maxRadius: number;
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
      <p class="subtitle">微笑让雨落下，大笑让焰火绽放；用面部与声音两条感知链模拟直播互动。</p>
    </section>

    <section class="stage-card">
      <div class="stage-wrap">
        <video id="camera" autoplay muted playsinline></video>
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
const effectRings: EffectRing[] = [];
let smileEffectMix = 0;
let laughEffectMix = 0;

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
  const cheekSquint = (
    getBlendshapeScore(result, 'cheekSquintLeft') +
    getBlendshapeScore(result, 'cheekSquintRight')
  ) / 2;
  const eyeSquint = (
    getBlendshapeScore(result, 'eyeSquintLeft') +
    getBlendshapeScore(result, 'eyeSquintRight')
  ) / 2;
  const mouthPress = (
    getBlendshapeScore(result, 'mouthPressLeft') +
    getBlendshapeScore(result, 'mouthPressRight')
  ) / 2;
  const mouthRoll = (
    getBlendshapeScore(result, 'mouthRollLower') +
    getBlendshapeScore(result, 'mouthRollUpper')
  ) / 2;
  const mouthPucker = getBlendshapeScore(result, 'mouthPucker');
  const lipPress = Math.max(mouthPress, mouthRoll, mouthPucker);

  // Keep enough recent history to remove one-frame spikes, but use a faster
  // response than the previous 0.20 weighting for live-stream-like feedback.
  smoothedSmile = smoothedSmile * 0.66 + smileScore * 0.34;
  smoothedJaw = smoothedJaw * 0.66 + jawOpen * 0.34;
  smoothedCheekSquint = smoothedCheekSquint * 0.66 + cheekSquint * 0.34;
  smoothedEyeSquint = smoothedEyeSquint * 0.66 + eyeSquint * 0.34;
  smoothedLipPress = smoothedLipPress * 0.66 + lipPress * 0.34;
  return {
    faceDetected: hasFace,
    smileScore: smoothedSmile,
    jawOpen: smoothedJaw,
    lipPress: smoothedLipPress,
    inferenceMs,
    lastUpdatedAt: performance.now(),
    headBounds: hasFace ? getHeadBounds(result.faceLandmarks[0], rect) : null,
  };
}

function isLipPressing(now: number): boolean {
  // The press signal can alternate with mouthSmile while the user holds the
  // same gesture. Latch strong evidence briefly so those alternating frames
  // do not make the rain flash on and off.
  const hasStrongEvidence = smoothedJaw <= 0.16 &&
    smoothedLipPress >= LIP_PRESS_ENTER &&
    smoothedLipPress >= smoothedSmile * 0.65;
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
    (faceSupport >= 0.04 || smoothedSmile >= 0.52);
  const expressiveVisualLaugh = smoothedSmile >= 0.72 &&
    smoothedJaw >= 0.10 &&
    faceSupport >= 0.035;
  const audioAssistedLaugh = audioSupport &&
    smoothedSmile >= 0.58 &&
    smoothedJaw >= 0.08;
  return standardVisualLaugh || expressiveVisualLaugh || audioAssistedLaugh;
}

function desiredInteractionState(now: number): InteractionState {
  if (!latestFeatures.faceDetected) return 'IDLE';
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
    const expressiveExit = smoothedSmile >= 0.58 &&
      smoothedJaw >= 0.08 &&
      (faceSupport >= 0.025 || audioSupport);
    const standardExit = smoothedSmile >= LAUGH_EXIT_SMILE &&
      smoothedJaw >= LAUGH_EXIT_JAW &&
      (exitSupport || strongMouthFallback);
    if (standardExit || expressiveExit) {
      return 'LAUGH';
    }
  }
  if (laughDetected) return 'LAUGH';

  // Lip pressing can lift the mouth corners and inflate mouthSmile. Treat it
  // as a veto for the rain effect instead of raising the global smile
  // threshold, which would make subtle real smiles impossible to detect.
  if (isLipPressing(now)) return 'IDLE';

  if (interactionState === 'SMILE') {
    if (now - stateEnteredAt < SMILE_MIN_DWELL_MS) return 'SMILE';
    if (smoothedSmile >= SMILE_EXIT && smoothedJaw <= SMILE_JAW_EXIT_MAX) return 'SMILE';
  }

  if (interactionState === 'OPEN_MOUTH' && smoothedJaw >= OPEN_MOUTH_EXIT) return 'OPEN_MOUTH';
  if (smoothedSmile >= SMILE_ENTER && smoothedJaw <= SMILE_JAW_ENTER_MAX) return 'SMILE';
  if (smoothedJaw >= OPEN_MOUTH_ENTER) return 'OPEN_MOUTH';
  return 'IDLE';
}

function transitionInteraction(next: InteractionState, now: number): void {
  if (next === interactionState) return;
  const previous = interactionState;
  interactionState = next;
  stateEnteredAt = now;

  // Rain particles used to keep their full multi-second lifetime after a
  // short false smile. Fade existing drops quickly when leaving SMILE so a
  // corrected IDLE state is also visually immediate.
  if (previous === 'SMILE' && next !== 'SMILE') {
    for (const particle of particles) {
      if (particle.kind === 'rain') particle.life = Math.min(particle.life, 0.28);
    }
    rainAccumulator = 0;
  }
  updateHud();

  if (next === 'LAUGH' && now - lastFireworkAt >= FIREWORK_COOLDOWN_MS) {
    spawnFirework();
    lastFireworkAt = now;
  }

  const messages: Record<InteractionState, string> = {
    IDLE: '状态稳定：保持自然表情，效果会逐渐停下。',
    SMILE: '检测到微笑：雨幕已启动。',
    OPEN_MOUTH: '检测到张嘴：保持中性，不触发焰火。',
    LAUGH: '检测到大笑：焰火已触发，粒子会与头部边界互动。',
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
  if (desired !== candidateState) {
    candidateState = desired;
    candidateSince = now;
  }
  const holdMs = candidateState === 'LAUGH'
    ? LAUGH_HOLD_MS
    : candidateState === 'SMILE'
      ? SMILE_HOLD_MS
      : STATE_HOLD_MS;
  if (candidateState !== interactionState && now - candidateSince >= holdMs) {
    transitionInteraction(candidateState, now);
  }
}

function drawDebugFace(): void {
  if (!SHOW_DEBUG_LANDMARKS || !ctx || !latestLandmarks) return;
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
    vy: 220 + Math.random() * 220,
    size: 6 + Math.random() * 11,
    life: 2.4 + Math.random() * 0.8,
    maxLife: 3.2,
    hue: 188 + Math.random() * 35,
    alpha: 0.22 + Math.random() * 0.38,
  });
}

function spawnFirework(): void {
  const rect = video.getBoundingClientRect();
  const bounds = latestFeatures.headBounds;
  const originX = bounds?.cx ?? rect.width * 0.5;
  const originY = bounds ? Math.max(48, bounds.cy - bounds.ry * 1.35) : rect.height * 0.35;
  const count = 68;
  const hue = 28 + Math.random() * 100;
  effectRings.push({
    x: originX,
    y: originY,
    radius: 12,
    maxRadius: 150 + Math.random() * 90,
    life: 0.58,
    maxLife: 0.58,
    hue,
    alpha: 0.75,
  });
  if (effectRings.length > 8) effectRings.shift();
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
      prevX: originX,
      prevY: originY,
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
    rainAccumulator += deltaSeconds * 105;
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

    particle.prevX = particle.x;
    particle.prevY = particle.y;
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

function updateEffectMix(deltaSeconds: number): void {
  const response = Math.min(1, deltaSeconds * 8);
  const smileTarget = interactionState === 'SMILE' ? 1 : 0;
  const laughTarget = interactionState === 'LAUGH' ? 1 : 0;
  smileEffectMix += (smileTarget - smileEffectMix) * response;
  laughEffectMix += (laughTarget - laughEffectMix) * response;
}

function updateEffectRings(deltaSeconds: number): void {
  for (let index = effectRings.length - 1; index >= 0; index -= 1) {
    const ring = effectRings[index];
    ring.life -= deltaSeconds;
    if (ring.life <= 0) {
      effectRings.splice(index, 1);
      continue;
    }
    const progress = 1 - ring.life / ring.maxLife;
    ring.radius = 12 + (ring.maxRadius - 12) * progress;
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

  if (laughEffectMix > 0.01) {
    const bounds = latestFeatures.headBounds;
    const glowX = bounds?.cx ?? rect.width * 0.5;
    const glowY = bounds ? Math.max(40, bounds.cy - bounds.ry * 1.05) : rect.height * 0.3;
    const laughGlow = ctx.createRadialGradient(glowX, glowY, 0, glowX, glowY, rect.width * 0.62);
    laughGlow.addColorStop(0, `rgba(255, 204, 113, ${0.24 * laughEffectMix})`);
    laughGlow.addColorStop(0.42, `rgba(255, 100, 78, ${0.08 * laughEffectMix})`);
    laughGlow.addColorStop(1, 'rgba(255, 70, 55, 0)');
    ctx.save();
    ctx.globalCompositeOperation = 'screen';
    ctx.fillStyle = laughGlow;
    ctx.fillRect(0, 0, rect.width, rect.height);
    ctx.restore();
  }
}

function drawEffectRings(): void {
  if (!ctx) return;
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  for (const ring of effectRings) {
    const lifeRatio = Math.max(0, ring.life / ring.maxLife);
    ctx.globalAlpha = ring.alpha * Math.min(1, lifeRatio * 1.8);
    ctx.strokeStyle = `hsla(${ring.hue}, 100%, 74%, 1)`;
    ctx.shadowColor = `hsla(${ring.hue}, 100%, 68%, 1)`;
    ctx.shadowBlur = 14;
    ctx.lineWidth = 2.2 + lifeRatio * 2;
    ctx.beginPath();
    ctx.arc(ring.x, ring.y, ring.radius, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

function drawParticles(): void {
  if (!ctx) return;
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  for (const particle of particles) {
    const lifeRatio = Math.max(0, particle.life / particle.maxLife);
    ctx.globalAlpha = particle.alpha * Math.min(1, lifeRatio * 1.8);
    ctx.strokeStyle = `hsla(${particle.hue}, 100%, 78%, 1)`;
    ctx.fillStyle = `hsla(${particle.hue}, 100%, 70%, 1)`;
    if (particle.kind === 'rain') {
      ctx.lineWidth = 1.4;
      ctx.shadowColor = `hsla(${particle.hue}, 100%, 72%, 1)`;
      ctx.shadowBlur = 6;
      ctx.beginPath();
      ctx.moveTo(particle.x, particle.y);
      ctx.lineTo(particle.x - particle.vx * 0.012, particle.y - particle.vy * 0.022);
      ctx.stroke();
    } else {
      ctx.shadowColor = `hsla(${particle.hue}, 100%, 70%, 1)`;
      ctx.shadowBlur = 11;
      ctx.lineWidth = Math.max(1, particle.size * 0.65);
      ctx.beginPath();
      ctx.moveTo(particle.prevX ?? particle.x, particle.prevY ?? particle.y);
      ctx.lineTo(particle.x, particle.y);
      ctx.stroke();
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
  if (canvas.width !== Math.floor(rect.width * Math.min(window.devicePixelRatio || 1, MAX_CANVAS_DPR))) resizeCanvas();
  ctx?.clearRect(0, 0, rect.width, rect.height);
  sampleAudio(now);

  if (faceLandmarker && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && now - lastInferenceAt >= INFERENCE_INTERVAL_MS) {
    const inferenceStart = performance.now();
    const result = faceLandmarker.detectForVideo(video, now);
    const inferenceCompletedAt = performance.now();
    lastInferenceAt = inferenceCompletedAt;
    latestLandmarks = result.faceLandmarks[0] ?? null;
    latestFeatures = readFeatures(result, inferenceCompletedAt - inferenceStart, rect);
    updateInteraction(inferenceCompletedAt);
    updateHud();
  }

  const deltaSeconds = Math.min(0.05, (now - previousRenderAt) / 1000);
  previousRenderAt = now;
  updateEffectMix(deltaSeconds);
  updateParticles(deltaSeconds, rect);
  updateEffectRings(deltaSeconds);
  drawStageAtmosphere(rect);
  drawEffectRings();
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
  effectRings.length = 0;
  smileEffectMix = 0;
  laughEffectMix = 0;
  smoothedSmile = 0;
  smoothedJaw = 0;
  smoothedCheekSquint = 0;
  smoothedEyeSquint = 0;
  smoothedLipPress = 0;
  lastStrongLipPressAt = -Infinity;
  interactionState = 'IDLE';
  candidateState = 'IDLE';
  stateEnteredAt = 0;
  previousRenderAt = performance.now();
  latestFeatures = {
    faceDetected: false,
    smileScore: 0,
    jawOpen: 0,
    lipPress: 0,
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
