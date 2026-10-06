import './style.css';
import { FaceLandmarker, FilesetResolver, type FaceLandmarkerResult } from '@mediapipe/tasks-vision';
import { ProceduralFireworkSystem } from './procedural-firework';

const MODEL_URL = '/assets/face_landmarker.task';
const WASM_ROOT = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
const INFERENCE_INTERVAL_MS = 1000 / 10;
const RENDER_INTERVAL_MS = 1000 / 60;
const MAX_CANVAS_DPR = 1.5;
const URL_PARAMS = new URLSearchParams(window.location.search);
const SHOW_DEBUG_LANDMARKS = URL_PARAMS.has('debug');
const PREVIEW_MODE = URL_PARAMS.get('preview');
const PREVIEW_FIREWORK = PREVIEW_MODE === 'firework' || PREVIEW_MODE === 'collision';
const PREVIEW_HEAD_SWEEP = PREVIEW_MODE === 'collision';
const PREVIEW_RAIN = PREVIEW_MODE === 'rain';
const IS_WEBKIT_BROWSER = /AppleWebKit/i.test(navigator.userAgent) && !/(Chrome|Chromium|Edg|OPR|Android)/i.test(navigator.userAgent);
const CAMERA_PERFORMANCE_GRACE_MS = 8000;
const CAMERA_POOR_WINDOWS_BEFORE_DOWNGRADE = 4;
const WEBKIT_RAIN_FRAME_INTERVAL_MS = 1000 / 20;
const WEBKIT_RAIN_WIDTH = 480;

const CAMERA_PROFILES = [
  { tier: 'high', label: '1080p', width: 1920, height: 1080, frameRate: 30 },
  { tier: 'balanced', label: '720p', width: 1280, height: 720, frameRate: 30 },
  { tier: 'efficient', label: '480p', width: 640, height: 480, frameRate: 30 },
] as const;

type CameraProfile = (typeof CAMERA_PROFILES)[number];

interface DevicePerformanceNavigator extends Navigator {
  deviceMemory?: number;
  connection?: {
    saveData?: boolean;
  };
}

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
const FRONTAL_NEUTRAL_SMILE_MAX = 0.13;
const FRONTAL_NEUTRAL_LIP_PRESS_MIN = 0.085;
const LAUGH_ENTER_SMILE = 0.34;
const LAUGH_ENTER_JAW = 0.30;
const LAUGH_EXIT_SMILE = 0.25;
const LAUGH_EXIT_JAW = 0.18;
const LAUGH_SMILE_TO_JAW_RATIO = 0.80;
const SIDE_FACE_START_RAD = 0.16;
const SIDE_FACE_FULL_RAD = 0.56;
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
  faceTurn: number;
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

interface RainRipple {
  x: number;
  y: number;
  radius: number;
  maxRadius: number;
  life: number;
  maxLife: number;
  alpha: number;
}

const app = document.querySelector<HTMLDivElement>('#app');
if (!app) throw new Error('App root not found');
document.documentElement.classList.toggle(
  'webkit-rain-blend',
  IS_WEBKIT_BROWSER,
);

app.innerHTML = `
  <div class="app-frame">
    <header class="topbar">
      <a class="brand" href="#" aria-label="TikTok Studio 首页">
        <span class="brand-mark" aria-hidden="true"><img src="/assets/tiktok-live-logo.png" alt="" width="96" height="96" /></span>
        <span>TikTok <b>Studio</b></span>
      </a>
      <div class="topbar-actions">
        <img class="profile-avatar" src="/assets/profile-zijie.png" alt="子杰的头像" width="28" height="28" /><span class="profile-name">子杰</span>
        <button class="icon-button" type="button" aria-label="帮助"><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M9.6 9a2.5 2.5 0 0 1 4.8 1c0 1.7-2.4 2-2.4 3.5M12 17.5h.01"/></svg></button>
        <button class="icon-button" type="button" aria-label="菜单"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg></button>
      </div>
    </header>

    <main class="shell">
      <div class="broadcast-grid">
        <aside class="scene-sidebar" aria-label="场景与来源">
          <section class="sidebar-section">
            <div class="sidebar-heading"><h2>场景</h2><button type="button" aria-label="新增场景">＋</button></div>
            <button class="scene-item active" type="button">
              <span class="scene-thumb"><i></i><b>AR</b></span>
              <span><strong>互动直播间</strong><small>雨幕与烟花</small></span>
              <em aria-hidden="true">•••</em>
            </button>
          </section>
          <section class="sidebar-section sources-section">
            <div class="sidebar-heading"><h2>来源</h2><button type="button" aria-label="新增来源">＋</button></div>
            <button class="source-item active" type="button">
              <span class="source-icon camera-source"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 10.5 20 8v8l-5-2.5v-3Z"/><rect x="3" y="6" width="12" height="12" rx="3"/></svg></span>
              <span><strong>摄像头</strong><small>FaceTime HD Camera</small></span>
              <svg class="eye-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12Z"/><circle cx="12" cy="12" r="2.5"/></svg>
            </button>
            <button class="source-item" type="button">
              <span class="source-icon effect-source"><img src="/assets/badge-ar.svg" alt="" width="20" height="20" /></span>
              <span><strong>AR 特效</strong><small>表情驱动 · 已启用</small></span>
              <svg class="eye-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12Z"/><circle cx="12" cy="12" r="2.5"/></svg>
            </button>
          </section>
          <div class="sidebar-footer"><button type="button"><span>＋</span>添加来源</button><button type="button">清除</button></div>
        </aside>

        <section id="live-stage" class="stage-card" aria-label="直播画面与控制">
          <div class="stage-header">
            <h1>子杰 is LIVE</h1>
            <div class="orientation-switch" role="group" aria-label="直播画面方向">
              <button class="active" type="button" data-orientation="landscape" aria-pressed="true">横屏</button>
              <button type="button" data-orientation="portrait" aria-pressed="false">竖屏</button>
            </div>
          </div>
          <div class="stage-wrap">
            <video id="camera" autoplay muted playsinline></video>
            <div class="rain-grade" aria-hidden="true"></div>
            <div class="rain-footage-layer" aria-hidden="true">
              <video id="rain-footage" class="rain-footage" src="/assets/rain-overlay-candidate.mp4" muted loop playsinline preload="metadata"></video>
            </div>
            <canvas id="rain-composite" class="rain-composite" aria-hidden="true"></canvas>
            <canvas id="overlay"></canvas><div class="stage-shade"></div>
            <div id="stage-message" class="stage-message">
              <div class="message-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M15 10.5 19.5 8v8L15 13.5v-3Z"/><rect x="3" y="6" width="12" height="12" rx="3"/></svg></div>
              <strong>开启直播预览</strong><span>允许摄像头后即可预览实时特效。</span>
            </div>
          </div>
          <div class="controls">
            <button id="start-button" class="primary-button">开始体验</button>
            <button id="reset-button" class="ghost-button" disabled>重置</button>
            <button id="audio-button" class="ghost-button" disabled aria-pressed="false">开启声音辅助</button>
          </div>
        </section>

        <aside id="interaction-panel" class="interaction-panel" aria-label="直播互动区">
          <section class="viewer-panel">
            <div class="panel-heading"><h2>观众列表</h2><button type="button" aria-label="弹出观众列表"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 3h7v7M10 14 21 3M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5"/></svg></button></div>
            <div class="empty-viewers" aria-label="暂无观众"></div>
          </section>
          <section class="live-chat-panel">
            <div class="panel-heading"><h2>直播聊天</h2><button type="button" aria-label="弹出直播聊天"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 3h7v7M10 14 21 3M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5"/></svg></button></div>
            <div id="chat-feed" class="chat-feed" aria-label="直播消息" aria-live="polite">
              <div class="chat-message"><span class="chat-avatar cyan">A</span><p><span>Hello host</span></p></div>
              <div class="chat-message"><span class="chat-avatar violet">M</span><p><span>Great stream</span></p></div>
            </div>
            <form id="chat-form" class="chat-composer">
              <label class="visually-hidden" for="chat-input">直播消息</label>
              <input id="chat-input" type="text" maxlength="200" autocomplete="off" placeholder="输入消息..." />
              <button id="chat-send" type="submit" disabled>发送</button>
            </form>
          </section>
        </aside>
      </div>

      <section class="runtime-data" aria-hidden="true">
        <span id="status-pill">未启动</span><span id="face-value">—</span><span id="smile-value">0.00</span><span id="jaw-value">0.00</span><span id="lip-press-value">0.00</span><span id="inference-value">—</span><span id="fps-value">0</span><span id="effect-value">IDLE</span><span id="audio-value">OFF</span><span id="audio-status" data-enabled="false">声音辅助关闭</span><span id="hint"></span>
      </section>
    </main>
  </div>
`;

const video = document.querySelector<HTMLVideoElement>('#camera')!;
const rainFootage = document.querySelector<HTMLVideoElement>('#rain-footage')!;
const rainComposite = document.querySelector<HTMLCanvasElement>('#rain-composite')!;
const rainCompositeContext = rainComposite.getContext('2d');
const stageWrap = document.querySelector<HTMLDivElement>('.stage-wrap')!;
const canvas = document.querySelector<HTMLCanvasElement>('#overlay')!;
const ctx = canvas.getContext('2d');
const startButton = document.querySelector<HTMLButtonElement>('#start-button')!;
const resetButton = document.querySelector<HTMLButtonElement>('#reset-button')!;
const audioButton = document.querySelector<HTMLButtonElement>('#audio-button')!;
const liveStage = document.querySelector<HTMLElement>('#live-stage')!;
const orientationButtons = [...document.querySelectorAll<HTMLButtonElement>('.orientation-switch button')];
const chatFeed = document.querySelector<HTMLDivElement>('#chat-feed')!;
const chatForm = document.querySelector<HTMLFormElement>('#chat-form')!;
const chatInput = document.querySelector<HTMLInputElement>('#chat-input')!;
const chatSend = document.querySelector<HTMLButtonElement>('#chat-send')!;
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
const rainFrameSampler = document.createElement('canvas');
const rainFrameSamplerContext = rainFrameSampler.getContext('2d');

let faceLandmarker: FaceLandmarker | null = null;
let faceLandmarkerLoadPromise: Promise<FaceLandmarker> | null = null;
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
  faceTurn: 0,
  inferenceMs: 0,
  lastUpdatedAt: 0,
  headBounds: null,
};
let frameCount = 0;
let lastFpsAt = performance.now();
let renderFps = 0;
let cameraQualityIndex = 1;
let cameraStartedAt = 0;
let cameraPoorPerformanceWindows = 0;
let cameraConstraintPending = false;
let lastWebKitRainFrameAt = -Infinity;
let previousRenderAt = performance.now();
let lastVisualFrameAt = -Infinity;
let rainAccumulator = 0;
let splashAccumulator = 0;
const particles: Particle[] = [];
const rainRipples: RainRipple[] = [];
let smileEffectMix = 0;
let laughEffectMix = 0;
let rainFootagePlaying = false;
let previewInitialized = false;
const proceduralFireworks = new ProceduralFireworkSystem();

function chooseInitialCameraQuality(): number {
  const deviceNavigator = navigator as DevicePerformanceNavigator;
  const cpuThreads = navigator.hardwareConcurrency || 0;
  const memoryGb = deviceNavigator.deviceMemory || 0;
  let score = 0;

  if (cpuThreads >= 8) score += 2;
  else if (cpuThreads > 0 && cpuThreads <= 2) score -= 2;

  if (memoryGb >= 8) score += 2;
  else if (memoryGb > 0 && memoryGb <= 2) score -= 2;

  if (deviceNavigator.connection?.saveData) score -= 3;
  if (score >= 2) return 0;
  if (score <= -2) return 2;
  return 1;
}

function cameraConstraints(profile: CameraProfile): MediaTrackConstraints {
  return {
    facingMode: 'user',
    width: { ideal: profile.width, max: profile.width },
    height: { ideal: profile.height, max: profile.height },
    frameRate: { ideal: profile.frameRate, max: profile.frameRate },
  };
}

function updateCameraDiagnostics(profile: CameraProfile): void {
  const settings = stream?.getVideoTracks()[0]?.getSettings();
  video.dataset.cameraTier = profile.tier;
  video.dataset.cameraRequested = `${profile.width}x${profile.height}@${profile.frameRate}`;
  video.dataset.cameraActual = `${settings?.width ?? video.videoWidth}x${settings?.height ?? video.videoHeight}@${Math.round(settings?.frameRate ?? 0)}`;
}

async function downgradeCameraQuality(): Promise<void> {
  if (!stream || cameraConstraintPending || cameraQualityIndex >= CAMERA_PROFILES.length - 1) return;
  const track = stream.getVideoTracks()[0];
  if (!track) return;

  cameraConstraintPending = true;
  const nextIndex = cameraQualityIndex + 1;
  const nextProfile = CAMERA_PROFILES[nextIndex];
  try {
    await track.applyConstraints(cameraConstraints(nextProfile));
    cameraQualityIndex = nextIndex;
    cameraStartedAt = performance.now();
    cameraPoorPerformanceWindows = 0;
    updateCameraDiagnostics(nextProfile);
    resizeCanvas();
    hint.textContent = `检测到设备运行压力，摄像头已自动调整为 ${nextProfile.label}。`;
  } catch (error) {
    console.warn('Unable to lower camera resolution automatically.', error);
  } finally {
    cameraConstraintPending = false;
  }
}

function evaluateCameraPerformance(now: number): void {
  if (!stream || uiState !== 'running' || document.visibilityState !== 'visible') return;
  if (cameraQualityIndex >= CAMERA_PROFILES.length - 1 || now - cameraStartedAt < CAMERA_PERFORMANCE_GRACE_MS) return;

  const inferenceIsSlow = latestFeatures.lastUpdatedAt > 0 && latestFeatures.inferenceMs > 65;
  const renderingIsSlow = renderFps > 0 && renderFps < 42;
  cameraPoorPerformanceWindows = inferenceIsSlow || renderingIsSlow
    ? cameraPoorPerformanceWindows + 1
    : Math.max(0, cameraPoorPerformanceWindows - 1);

  if (cameraPoorPerformanceWindows >= CAMERA_POOR_WINDOWS_BEFORE_DOWNGRADE) {
    void downgradeCameraQuality();
  }
}

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
  const rect = stageWrap.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, MAX_CANVAS_DPR);
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  ctx?.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function updateWebKitRainComposite(now: number): void {
  if (
    !IS_WEBKIT_BROWSER ||
    !rainCompositeContext ||
    !rainFrameSamplerContext ||
    !rainFootagePlaying ||
    rainFootage.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
    now - lastWebKitRainFrameAt < WEBKIT_RAIN_FRAME_INTERVAL_MS
  ) return;

  lastWebKitRainFrameAt = now;
  const sourceRatio = rainFootage.videoHeight > 0
    ? rainFootage.videoHeight / rainFootage.videoWidth
    : 9 / 16;
  const width = WEBKIT_RAIN_WIDTH;
  const height = Math.max(1, Math.round(width * sourceRatio));
  if (rainFrameSampler.width !== width || rainFrameSampler.height !== height) {
    rainFrameSampler.width = width;
    rainFrameSampler.height = height;
    rainComposite.width = width;
    rainComposite.height = height;
  }

  rainFrameSamplerContext.drawImage(rainFootage, 0, 0, width, height);
  const frame = rainFrameSamplerContext.getImageData(0, 0, width, height);
  for (let index = 0; index < frame.data.length; index += 4) {
    const luminance = (
      frame.data[index] * 0.2126 +
      frame.data[index + 1] * 0.7152 +
      frame.data[index + 2] * 0.0722
    ) / 255;
    const keyedLuminance = Math.max(0, Math.min(1, (luminance * 0.59 - 0.5) * 6 + 0.5));
    frame.data[index] = 238;
    frame.data[index + 1] = 246;
    frame.data[index + 2] = 250;
    frame.data[index + 3] = Math.round(keyedLuminance * 255);
  }
  rainCompositeContext.putImageData(frame, 0, 0);
}

function setStageOrientation(orientation: 'landscape' | 'portrait'): void {
  liveStage.dataset.orientation = orientation;
  for (const button of orientationButtons) {
    const isActive = button.dataset.orientation === orientation;
    button.classList.toggle('active', isActive);
    button.setAttribute('aria-pressed', String(isActive));
  }
  requestAnimationFrame(resizeCanvas);
}

function updateChatSendState(): void {
  chatSend.disabled = chatInput.value.trim().length === 0;
}

function appendChatMessage(message: string): void {
  const item = document.createElement('div');
  item.className = 'chat-message chat-message-own';

  const avatar = document.createElement('img');
  avatar.className = 'chat-avatar chat-avatar-photo';
  avatar.src = '/assets/profile-zijie.png';
  avatar.alt = '';
  avatar.width = 27;
  avatar.height = 27;

  const text = document.createElement('p');
  const content = document.createElement('span');
  content.textContent = message;
  text.append(content);
  item.append(avatar, text);
  chatFeed.append(item);
  chatFeed.scrollTop = chatFeed.scrollHeight;
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
  canvas.dataset.faceTurn = latestFeatures.faceTurn.toFixed(3);
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
    audioButton.textContent = '开启声音辅助';
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
  audioButton.textContent = '开启声音辅助';
  audioButton.disabled = uiState !== 'running';
  updateAudioStatus('声音辅助关闭 · 不读取麦克风', false);
  updateHud();
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function getBlendshapeScore(result: FaceLandmarkerResult, name: string, faceIndex = 0): number {
  const categories = result.faceBlendshapes?.[faceIndex]?.categories ?? [];
  return categories.find((category) => category.categoryName === name)?.score ?? 0;
}

function smoothResponsive(current: number, sample: number, riseWeight = 0.48, fallWeight = 0.30): number {
  const weight = sample > current ? riseWeight : fallWeight;
  return current + (sample - current) * weight;
}

function getFaceTurn(result: FaceLandmarkerResult, faceIndex: number, landmarks: FaceLandmarkerResult['faceLandmarks'][number] | null): number {
  // FaceLandmarker already provides a pose matrix. The absolute yaw is all we
  // need here: left and right profile views should receive identical handling.
  // For the usual row- or column-major rotation layout, indices 2 and 10 form
  // the yaw sine/cosine pair (the sign changes, but its magnitude does not).
  const matrix = result.facialTransformationMatrixes?.[faceIndex]?.data;
  if (matrix && matrix.length >= 11 && Number.isFinite(matrix[2]) && Number.isFinite(matrix[10])) {
    return clamp(Math.abs(Math.atan2(matrix[2], matrix[10])), 0, Math.PI / 2);
  }
  if (!landmarks?.[1] || !landmarks[234] || !landmarks[454]) return 0;

  // Safe fallback for implementations that omit the pose matrix. As the head
  // turns, the nose moves away from the midpoint of the two face sides.
  const nose = landmarks[1];
  const leftTemple = landmarks[234];
  const rightTemple = landmarks[454];
  const faceSpan = Math.max(0.001, Math.abs(rightTemple.x - leftTemple.x));
  const faceCenterX = (leftTemple.x + rightTemple.x) / 2;
  const normalizedOffset = Math.abs(nose.x - faceCenterX) / faceSpan;
  return clamp(normalizedOffset * 1.35, 0, 1) * (Math.PI / 2);
}

function getSideFaceWeight(faceTurn: number): number {
  return clamp((faceTurn - SIDE_FACE_START_RAD) / (SIDE_FACE_FULL_RAD - SIDE_FACE_START_RAD), 0, 1);
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
    // Face landmarks stop at the forehead, while users perceive hair as part
    // of the physical head. Shift the collider upward and extend its crown so
    // sparks do not visibly travel through hair before reacting.
    cy: (minY + maxY) / 2 - height * 0.12,
    rx: Math.max(48, width * 0.65),
    ry: Math.max(60, height * 0.74),
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
  const faceTurn = getFaceTurn(result, faceIndex, landmarks);
  const sideFaceWeight = getSideFaceWeight(faceTurn);

  // On a small face the dedicated mouthSmile blendshape often collapses
  // toward zero. mouthDimple survives at lower pixel density more reliably,
  // so use it only as distant-face support instead of lowering every global
  // smile threshold and reintroducing the lip-press false positive.
  // Weight the weaker side as well as the average. This preserves a real
  // bilateral smile but suppresses the one-sided false positive visible when
  // the user's head is tilted and just one mouth corner appears raised.
  const bilateralSmile = smileScore * 0.68 + symmetricSmile * 0.32;
  const bilateralDimple = mouthDimple * 0.68 + symmetricDimple * 0.32;
  // A profile view hides one mouth corner, so the weaker-side weighting that
  // protects a frontal face from smirk false positives becomes destructive.
  // Blend toward the visible/stronger side only in proportion to measured yaw;
  // frontal thresholds therefore remain unchanged.
  const profileSmile = Math.max(leftSmile, rightSmile) * 0.68 + smileScore * 0.32;
  const profileDimple = Math.max(leftDimple, rightDimple) * 0.68 + mouthDimple * 0.32;
  const poseAdjustedSmile = bilateralSmile + Math.max(0, profileSmile - bilateralSmile) * sideFaceWeight;
  const poseAdjustedDimple = bilateralDimple + Math.max(0, profileDimple - bilateralDimple) * sideFaceWeight;
  const smileEvidence = isDistantFace
    ? Math.max(poseAdjustedSmile, poseAdjustedDimple * 0.82)
    : poseAdjustedSmile;

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
    faceTurn,
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
  const sideFaceWeight = getSideFaceWeight(latestFeatures.faceTurn);
  const widthRelaxation = sideFaceWeight * 0.060;
  const geometryRelaxation = 1 - sideFaceWeight * 0.54;
  const pressEnter = (isDistantFace ? LIP_PRESS_ENTER - 0.015 : LIP_PRESS_ENTER) + sideFaceWeight * 0.055;
  const geometryConfirmsSmile = latestFeatures.geometrySmile >= (isDistantFace ? 0.050 : 0.075) * geometryRelaxation &&
    latestFeatures.mouthWidthRatio >= (isDistantFace ? 0.355 : 0.385) - widthRelaxation;
  const geometrySuggestsSmile = latestFeatures.geometrySmile >= (isDistantFace ? 0.025 : 0.035) * geometryRelaxation &&
    latestFeatures.mouthWidthRatio >= (isDistantFace ? 0.335 : 0.36) - widthRelaxation;
  const pressDominates = smoothedLipPress >= smoothedSmile * ((isDistantFace ? 0.76 : 0.82) + sideFaceWeight * 0.40);
  const profileSmileFloor = Math.max(
    0.09,
    (isDistantFace ? 0.105 : 0.120) - sideFaceWeight * 0.025,
  );
  // With a turned head MediaPipe frequently labels the compressed/hidden lip
  // as mouthPress. A visible-side smile plus lifted, sufficiently wide geometry
  // is stronger evidence than that pose-induced press score.
  const profileSmileOverridesPress = sideFaceWeight >= 0.24 &&
    smoothedSmile >= profileSmileFloor &&
    smoothedJaw <= 0.12 &&
    smoothedLipPress <= 0.29 &&
    geometrySuggestsSmile;
  if (profileSmileOverridesPress) return false;
  const closedNarrowMismatch = !geometrySuggestsSmile &&
    latestFeatures.geometrySmile < 0.045 &&
    latestFeatures.geometryMouthOpen < 0.035 &&
    latestFeatures.mouthWidthRatio < (isDistantFace ? 0.34 : 0.365) - widthRelaxation &&
    smoothedLipPress >= (isDistantFace ? 0.045 : 0.055) + sideFaceWeight * 0.045 &&
    smoothedSmile < 0.44;
  // At close range the blendshape classifier is more reliable than the
  // corner-lift geometry fallback. A closed mouth with almost zero model
  // smile and a visible press must cancel rain even if perspective makes the
  // mouth corners look lifted. This is the exact failure visible in the
  // user's screenshot (smile 0.00, jaw 0.00, lip press 0.09).
  const directClosedPress = smoothedJaw <= 0.085 &&
    smoothedLipPress >= (isDistantFace ? 0.050 : CLOSED_LIP_PRESS_ENTER) &&
    smoothedSmile <= (isDistantFace ? 0.16 : 0.12) - sideFaceWeight * 0.025 &&
    !geometryConfirmsSmile;
  // MediaPipe can report a pursed expression as both mouthSmile and
  // mouthPress. Treat it as a hard veto only when the geometry still looks
  // narrow; a widened closed-mouth smile must remain eligible for rain.
  const hardClosedPress = smoothedJaw <= 0.055 &&
    smoothedLipPress >= 0.095 + sideFaceWeight * 0.055 &&
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
  // A yawn or a deliberately opened mouth can still raise mouthSmile because
  // its corners stretch. Every purely visual laugh path must retain a plausible
  // smile-to-jaw ratio; strong cheek/eye expression can rescue a real wide-open
  // laugh, but a neutral-eyed open mouth cannot trigger fireworks.
  const strongExpressiveOverride = smoothedSmile >= 0.62 && faceSupport >= 0.035;
  const hasVisualLaughShape = smileJawBalanced || strongExpressiveOverride;
  const standardVisualLaugh = smoothedSmile >= LAUGH_ENTER_SMILE &&
    smoothedJaw >= LAUGH_ENTER_JAW &&
    smileJawBalanced &&
    (faceSupport >= 0.02 || smoothedSmile >= 0.48);
  const broadOpenSmile = hasVisualLaughShape && smoothedSmile >= 0.50 && smoothedJaw >= 0.18;
  const geometryVisualLaugh = latestFeatures.geometryMouthOpen >= GEOMETRY_LAUGH_OPEN_ENTER &&
    hasVisualLaughShape &&
    smoothedSmile >= 0.26 &&
    (latestFeatures.geometrySmile >= 0.045 || faceSupport >= 0.018);
  const expressiveVisualLaugh = smoothedSmile >= 0.66 &&
    hasVisualLaughShape &&
    (smoothedJaw >= 0.08 || latestFeatures.geometryMouthOpen >= 0.045) &&
    faceSupport >= 0.025;
  const audioLaughShape = smoothedSmile >= smoothedJaw * 0.60 || strongExpressiveOverride;
  const audioAssistedLaugh = audioSupport &&
    audioLaughShape &&
    smoothedSmile >= 0.46 &&
    (smoothedJaw >= 0.07 || latestFeatures.geometryMouthOpen >= 0.04);
  return standardVisualLaugh || broadOpenSmile || geometryVisualLaugh || expressiveVisualLaugh || audioAssistedLaugh;
}

function hasIntentionalSmileEvidence(entering: boolean): boolean {
  const isDistantFace = Boolean(
    latestFeatures.headBounds && latestFeatures.headBounds.scale < DISTANT_FACE_SCALE,
  );
  const sideFaceWeight = getSideFaceWeight(latestFeatures.faceTurn);
  const baseSmileThreshold = isDistantFace
    ? entering ? DISTANT_SMILE_ENTER : DISTANT_SMILE_EXIT
    : entering ? SMILE_ENTER : SMILE_EXIT;
  // The visible-side score still settles around 0.12–0.14 in a real profile
  // smile. Lower only this pose-qualified threshold; the geometry checks below
  // remain mandatory, so a neutral or pursed profile cannot pass on score alone.
  const smileThreshold = Math.max(
    entering ? 0.075 : 0.045,
    baseSmileThreshold - sideFaceWeight * (entering ? 0.055 : 0.035),
  );
  const strongThreshold = isDistantFace
    ? entering ? DISTANT_SMILE_STRONG_ENTER : DISTANT_SMILE_STRONG_EXIT
    : entering ? SMILE_STRONG_ENTER : SMILE_STRONG_EXIT;
  const baseGeometryThreshold = isDistantFace
    ? entering ? DISTANT_GEOMETRY_SMILE_ENTER : DISTANT_GEOMETRY_SMILE_EXIT
    : entering ? GEOMETRY_SMILE_ENTER : GEOMETRY_SMILE_EXIT;
  const baseWidthThreshold = isDistantFace
    ? entering ? DISTANT_SMILE_WIDTH_ENTER : DISTANT_SMILE_WIDTH_EXIT
    : entering ? SMILE_WIDTH_ENTER : SMILE_WIDTH_EXIT;
  // Mouth width and the hidden corner's lift shrink under perspective. Apply a
  // bounded correction from measured yaw rather than weakening the frontal and
  // distant-face rules globally.
  const geometryThreshold = baseGeometryThreshold * (1 - sideFaceWeight * 0.54);
  const widthThreshold = baseWidthThreshold - sideFaceWeight * 0.060;
  const profileSmileThreshold = Math.max(
    entering ? 0.095 : 0.055,
    (isDistantFace ? entering ? 0.105 : 0.065 : entering ? 0.120 : 0.070) - sideFaceWeight * 0.025,
  );

  // Even a high classifier value needs a weak independent geometry check.
  // Borderline values need the stricter three-way agreement. This removes the
  // last single-signal path that could turn a head tilt or smirk into rain.
  const strongModelSignal = smoothedSmile >= strongThreshold &&
    smoothedLipPress <= (isDistantFace ? 0.10 : 0.08) + sideFaceWeight * 0.15 &&
    latestFeatures.geometrySmile >= geometryThreshold * 0.34 &&
    latestFeatures.mouthWidthRatio >= widthThreshold - 0.035;
  // A genuine closed-mouth smile can legitimately raise mouthPress a little
  // (the user's capture is smile 0.17 / press 0.12). Accept it when a softer
  // geometric shape check still shows lifted, widened corners; a real pout is
  // narrow and will fail this branch.
  const closedSmileSignal = smoothedSmile >= smileThreshold &&
    smoothedJaw <= 0.09 &&
    smoothedLipPress <= (isDistantFace ? 0.15 : 0.14) + sideFaceWeight * 0.13 &&
    latestFeatures.geometrySmile >= geometryThreshold * 0.48 &&
    latestFeatures.mouthWidthRatio >= widthThreshold - 0.025;
  const corroboratedSignal = smoothedSmile >= smileThreshold &&
    latestFeatures.geometrySmile >= geometryThreshold &&
    latestFeatures.mouthWidthRatio >= widthThreshold;
  const profileSmileSignal = sideFaceWeight >= 0.24 &&
    smoothedSmile >= profileSmileThreshold &&
    smoothedJaw <= 0.11 &&
    smoothedLipPress <= 0.29 &&
    latestFeatures.geometrySmile >= geometryThreshold * 0.56 &&
    latestFeatures.mouthWidthRatio >= widthThreshold - 0.012;
  return strongModelSignal || closedSmileSignal || corroboratedSignal || profileSmileSignal;
}

function desiredInteractionState(now: number): InteractionState {
  if (!latestFeatures.faceDetected) return 'IDLE';
  const sideFaceWeight = getSideFaceWeight(latestFeatures.faceTurn);
  // After a valid profile smile, returning to a neutral frontal face can leave
  // the smoothed score around 0.11 for several frames. That is below the entry
  // threshold but used to satisfy the looser exit hysteresis, keeping rain on.
  // Release it immediately only for a frontal, closed, lip-compressed mouth;
  // the profile-specific 0.12–0.14 smile path remains unaffected.
  const returnedToFrontalNeutral = interactionState === 'SMILE' &&
    sideFaceWeight < 0.24 &&
    smoothedSmile < FRONTAL_NEUTRAL_SMILE_MAX &&
    smoothedJaw <= 0.09 &&
    smoothedLipPress >= FRONTAL_NEUTRAL_LIP_PRESS_MIN;
  if (returnedToFrontalNeutral) return 'IDLE';
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
    const smileJawBalanced = smoothedSmile >= smoothedJaw * LAUGH_SMILE_TO_JAW_RATIO;
    const exitSupport = (faceSupport >= 0.03 && smileJawBalanced) || audioSupport;
    const strongMouthFallback = !audioEnabled && smoothedSmile >= 0.44 && smoothedJaw >= LAUGH_ENTER_JAW && smileJawBalanced;
    const expressiveExit = smileJawBalanced && smoothedSmile >= 0.48 &&
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
    faceTurnDegrees: (latestFeatures.faceTurn * 180 / Math.PI).toFixed(1),
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
  const rect = stageWrap.getBoundingClientRect();
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

function spawnProceduralFirework(mode: 'entry' | 'sustain'): void {
  const rect = stageWrap.getBoundingClientRect();
  proceduralFireworks.spawn(
    latestFeatures.headBounds,
    { width: rect.width, height: rect.height },
    mode,
  );
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
  stageWrap.style.setProperty('--rain-footage-opacity', (smileEffectMix * 0.58).toFixed(3));
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

function drawStageAtmosphere(rect: DOMRect): void {
  if (!ctx) return;

  if (smileEffectMix > 0.01) {
    const rainGlow = ctx.createLinearGradient(0, 0, 0, rect.height);
    rainGlow.addColorStop(0, `rgba(50, 196, 255, ${0.1 * smileEffectMix})`);
    rainGlow.addColorStop(0.52, `rgba(31, 107, 207, ${0.025 * smileEffectMix})`);
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
    evaluateCameraPerformance(now);
    if (PREVIEW_FIREWORK || PREVIEW_RAIN) updateHud();
  }

  const rect = stageWrap.getBoundingClientRect();
  if (PREVIEW_RAIN && !previewInitialized && rect.width > 0) {
    previewInitialized = true;
    stageMessage.hidden = true;
    stageMessage.classList.add('hidden');
    startButton.textContent = '体验中';
    uiState = 'running';
    interactionState = 'SMILE';
    stateEnteredAt = now;
    smileEffectMix = 1;
    stageWrap.classList.add('rain-active');
  }
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

  if (PREVIEW_HEAD_SWEEP && latestFeatures.headBounds) {
    const sweepProgress = now / 520;
    latestFeatures.headBounds = {
      ...latestFeatures.headBounds,
      cx: rect.width * 0.5 + Math.sin(sweepProgress) * rect.width * 0.18,
      cy: rect.height * 0.62 + Math.sin(sweepProgress * 0.54) * rect.height * 0.035,
      angle: Math.sin(sweepProgress * 0.72) * 0.10,
    };
  }

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
  if (uiState === 'running' && !PREVIEW_FIREWORK && !PREVIEW_RAIN) updateInteraction(performance.now());

  const deltaSeconds = Math.min(0.05, (now - previousRenderAt) / 1000);
  previousRenderAt = now;
  updateSustainedFireworks(now);
  updateEffectMix(deltaSeconds);
  updateWebKitRainComposite(now);
  updateParticles(deltaSeconds, rect);
  updateRainRipples(deltaSeconds);
  proceduralFireworks.update(deltaSeconds, latestFeatures.headBounds);
  if (PREVIEW_RAIN) canvas.dataset.activeParticles = String(particles.length);
  if (PREVIEW_FIREWORK) {
    canvas.dataset.collisionHits = String(proceduralFireworks.collisionCount);
    canvas.dataset.activeCollisionHits = String(proceduralFireworks.activeCollisionCount);
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

function loadFaceLandmarker(): Promise<FaceLandmarker> {
  if (!faceLandmarkerLoadPromise) {
    faceLandmarkerLoadPromise = createLandmarker().catch((error) => {
      faceLandmarkerLoadPromise = null;
      throw error;
    });
  }
  return faceLandmarkerLoadPromise;
}

async function startExperience(): Promise<void> {
  if (uiState === 'loading' || uiState === 'running') return;
  setUiState('loading', '正在请求摄像头权限并加载 Face Landmarker…');
  startButton.disabled = true;
  const landmarkerPromise = loadFaceLandmarker();

  try {
    cameraQualityIndex = chooseInitialCameraQuality();
    const initialCameraProfile = CAMERA_PROFILES[cameraQualityIndex];
    stream = await navigator.mediaDevices.getUserMedia({
      video: cameraConstraints(initialCameraProfile),
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    cameraStartedAt = performance.now();
    cameraPoorPerformanceWindows = 0;
    updateCameraDiagnostics(initialCameraProfile);
    resizeCanvas();
    stageMessage.querySelector('strong')!.textContent = '摄像头已开启';
    stageMessage.querySelector('span')!.textContent = '正在初始化人脸识别，首次打开可能需要几秒钟…';
    faceLandmarker = await landmarkerPromise;
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
  faceLandmarkerLoadPromise = null;
  latestLandmarks = null;
  particles.length = 0;
  rainRipples.length = 0;
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
  lastWebKitRainFrameAt = -Infinity;
  rainCompositeContext?.clearRect(0, 0, rainComposite.width, rainComposite.height);
  cameraQualityIndex = 1;
  cameraStartedAt = 0;
  cameraPoorPerformanceWindows = 0;
  cameraConstraintPending = false;
  delete video.dataset.cameraTier;
  delete video.dataset.cameraRequested;
  delete video.dataset.cameraActual;
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
    faceTurn: 0,
    inferenceMs: 0,
    lastUpdatedAt: 0,
    headBounds: null,
  };
  updateHud();
  stageMessage.classList.remove('hidden');
  stageMessage.querySelector('strong')!.textContent = '开启直播预览';
  stageMessage.querySelector('span')!.textContent = '允许摄像头后即可预览表情驱动的实时特效。';
  resetButton.disabled = true;
  startButton.disabled = false;
  startButton.textContent = '开始体验';
  audioButton.disabled = true;
  setUiState('idle', '技术验证目标：确认表情状态可以稳定驱动不同效果。');
}

void loadFaceLandmarker().catch((error) => {
  console.warn('Face Landmarker preload failed; it will retry when the experience starts.', error);
});

startButton.addEventListener('click', () => void startExperience());
resetButton.addEventListener('click', resetExperience);
audioButton.addEventListener('click', () => void enableAudioAssist());
for (const button of orientationButtons) {
  button.addEventListener('click', () => {
    const orientation = button.dataset.orientation;
    if (orientation === 'landscape' || orientation === 'portrait') setStageOrientation(orientation);
  });
}
chatInput.addEventListener('input', updateChatSendState);
chatForm.addEventListener('submit', (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;
  appendChatMessage(message);
  chatInput.value = '';
  updateChatSendState();
  chatInput.focus();
});
window.addEventListener('resize', resizeCanvas);
setStageOrientation('landscape');
updateChatSendState();
setUiState('idle');
if (PREVIEW_FIREWORK || PREVIEW_RAIN) render(performance.now());
else requestAnimationFrame(render);
