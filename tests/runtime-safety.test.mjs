import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';

// Execute the production lifecycle and scheduler without loading the app,
// MediaPipe, real DOM, network, camera, or microphone.
const source = readFileSync(new URL('../src/main.ts', import.meta.url), 'utf8');
const tree = ts.createSourceFile('main.ts', source, ts.ScriptTarget.Latest, true);
const functions = [
  'startExperience', 'resetExperience', 'loadFaceLandmarker',
  'enableAudioAssist', 'disableAudioAssist', 'updateAudioStatus', 'setUiState',
  'render', 'chooseInitialCameraQuality', 'cameraConstraints',
  'updateCameraDiagnostics', 'evaluateCameraPerformance', 'downgradeCameraQuality',
];
const constants = new Set([
  'CAMERA_PROFILES', 'CAMERA_PERFORMANCE_GRACE_MS', 'CAMERA_POOR_WINDOWS_BEFORE_DOWNGRADE',
  'RENDER_INTERVAL_MS', 'INFERENCE_INTERVAL_MS', 'MAX_CANVAS_DPR',
]);
const stateStart = tree.statements.findIndex(node => ts.isVariableStatement(node) &&
  node.declarationList.declarations.some(declaration => declaration.name.getText(tree) === 'faceLandmarker'));
const stateEnd = tree.statements.findIndex(node => ts.isFunctionDeclaration(node) && node.name.text === 'chooseInitialCameraQuality');
assert.ok(stateStart >= 0 && stateEnd > stateStart);
const selected = tree.statements.filter((node, index) =>
  (index >= stateStart && index < stateEnd) ||
  (ts.isFunctionDeclaration(node) && functions.includes(node.name?.text)) ||
  (ts.isVariableStatement(node) && node.declarationList.declarations.some(declaration => constants.has(declaration.name.getText(tree)))));
const compiled = ts.transpileModule(selected.map(node => node.getText(tree)).join('\n'), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
}).outputText;

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function mediaStream() {
  const track = {
    stops: 0, readyState: 'live', constraints: [],
    stop() { this.stops += 1; this.readyState = 'ended'; },
    getSettings() { return { width: 1280, height: 720, frameRate: 30 }; },
    async applyConstraints(value) { this.constraints.push(value); },
  };
  return { track, getTracks: () => [track], getVideoTracks: () => [track] };
}

function element() {
  const classes = new Set();
  const children = new Map();
  return {
    disabled: false, textContent: '', dataset: {}, attributes: {}, style: { setProperty() {} },
    classList: { add: value => classes.add(value), remove: value => classes.delete(value), contains: value => classes.has(value) },
    setAttribute(name, value) { this.attributes[name] = value; },
    querySelector(name) { if (!children.has(name)) children.set(name, element()); return children.get(name); },
    getBoundingClientRect: () => ({ width: 100, height: 60 }),
  };
}

function harness() {
  let now = 0;
  const cameras = [], microphones = [], models = [], audioPlans = [];
  const frames = [], scheduled = [], contexts = [], mediaRequests = [], warnings = [];
  const video = { ...element(), srcObject: null, readyState: 0, pauses: 0, async play() {}, pause() { this.pauses += 1; } };
  const sandbox = {
    console: { error() {}, warn(...args) { warnings.push(args); } }, DOMException,
    performance: { now: () => now }, document: { visibilityState: 'visible' },
    window: { devicePixelRatio: 1 }, HTMLMediaElement: { HAVE_CURRENT_DATA: 2 },
    navigator: { hardwareConcurrency: 4, mediaDevices: { getUserMedia(options) {
      mediaRequests.push(options);
      const queue = options.audio ? microphones : cameras;
      assert.ok(queue.length, 'unexpected media request');
      return Promise.resolve(queue.shift());
    } } },
    createLandmarker() { assert.ok(models.length, 'unexpected model request'); return Promise.resolve(models.shift()); },
    requestAnimationFrame(callback) { scheduled.push(callback); },
    video, canvas: { ...element(), width: 100 }, ctx: null,
    rainFootage: { currentTime: 0, pause() {} }, rainCompositeContext: null, rainComposite: {},
    PREVIEW_FIREWORK: false, PREVIEW_RAIN: false, PREVIEW_HEAD_SWEEP: false,
    ProceduralFireworkSystem: class {
      clear() {} elapseStalledTime() {}
      update(delta, bounds, elapsed) { frames.push({ now, delta, elapsed }); }
    },
    AudioContext: class {
      constructor() {
        this.plan = audioPlans.shift() ?? {};
        this.state = this.plan.resume ? 'suspended' : 'running';
        this.closes = 0;
        this.sources = 0;
        this.analyser = { disconnects: 0, disconnect() { this.disconnects += 1; } };
        contexts.push(this);
      }
      resume() { return this.plan.resume; }
      close() { this.closes += 1; this.state = 'closed'; return Promise.resolve(); }
      createMediaStreamSource() { this.sources += 1; return { connect() {} }; }
      createAnalyser() { return this.analyser; }
    },
  };
  for (const name of ['startButton', 'resetButton', 'audioButton', 'audioStatus', 'statusPill', 'hint', 'stageMessage', 'stageWrap']) sandbox[name] = element();
  // Match the disabled controls in the initial production HTML.
  sandbox.resetButton.disabled = true;
  sandbox.audioButton.disabled = true;
  for (const name of ['resizeCanvas', 'updateHud', 'sampleAudio', 'updateInteraction', 'updateSustainedFireworks', 'updateEffectMix', 'updateWebKitRainComposite', 'updateParticles', 'updateRainRipples', 'drawStageAtmosphere', 'drawRainRipples', 'drawParticles', 'drawDebugFace']) sandbox[name] = () => {};
  const context = vm.createContext(sandbox);
  vm.runInContext(`${compiled}\nglobalThis.api = { ${functions.join(',')}, state: () => ({ uiState, stream, faceLandmarker, faceLandmarkerLoadPromise, audioStream, audioContext, audioEnabled, audioAnalyser, audioSamples, renderFps, cameraQualityIndex, cameraPoorPerformanceWindows, cameraConstraintPending, cameraStartedAt }) };`, context);
  return {
    ...sandbox.api, sandbox, cameras, microphones, models, audioPlans, contexts, frames, scheduled, mediaRequests, warnings, video,
    tick(timestamp) { now = timestamp; sandbox.api.render(timestamp); },
  };
}

function faceModel() { return { closes: 0, close() { this.closes += 1; } }; }
async function start(h) {
  const camera = mediaStream(), model = faceModel();
  h.cameras.push(camera); h.models.push(model);
  await h.startExperience();
  assert.equal(h.state().uiState, 'running');
  return { camera, model };
}
async function flush() { for (let index = 0; index < 8; index += 1) await Promise.resolve(); }
function audioUi(h) {
  return {
    status: h.sandbox.audioStatus.textContent, enabled: h.sandbox.audioStatus.dataset.enabled,
    text: h.sandbox.audioButton.textContent, disabled: h.sandbox.audioButton.disabled,
    pressed: h.sandbox.audioButton.attributes['aria-pressed'], hint: h.sandbox.hint.textContent,
  };
}

function cameraUi(h) {
  const state = h.state();
  return {
    uiState: state.uiState, cameraQualityIndex: state.cameraQualityIndex,
    cameraStartedAt: state.cameraStartedAt, cameraPoorPerformanceWindows: state.cameraPoorPerformanceWindows,
    cameraConstraintPending: state.cameraConstraintPending,
    dataset: { ...h.video.dataset }, hint: h.sandbox.hint.textContent,
    resetDisabled: h.sandbox.resetButton.disabled, startDisabled: h.sandbox.startButton.disabled,
  };
}

for (const restart of [false, true]) {
  for (const outcome of ['resolve', 'reject']) {
    test(`model loading can be reset${restart ? ' and restarted' : ''}; obsolete ${outcome} cannot change UI`, async () => {
      const h = harness(), delayed = deferred(), camera = mediaStream(), model = faceModel();
      h.cameras.push(camera); h.models.push(delayed.promise);
      const pending = h.startExperience(); await flush();
      assert.equal(h.video.srcObject, camera);
      assert.equal(h.state().uiState, 'loading');
      assert.equal(h.sandbox.resetButton.disabled, false, 'reset must be available while the model is pending');
      assert.equal(h.sandbox.audioButton.disabled, true);
      h.resetExperience();
      assert.equal(camera.track.readyState, 'ended', 'reset must stop the camera before model settlement');
      assert.equal(h.video.srcObject, null);
      assert.equal(h.state().uiState, 'idle');
      const current = restart ? await start(h) : null;
      const ui = cameraUi(h);
      if (outcome === 'resolve') delayed.resolve(model);
      else delayed.reject(new Error('obsolete model failure'));
      await pending; await flush();
      assert.deepEqual(cameraUi(h), ui);
      assert.equal(h.video.srcObject, current?.camera ?? null);
      assert.equal(h.state().faceLandmarker, current?.model ?? null);
      if (current) assert.equal(current.camera.track.stops, 0);
      if (outcome === 'resolve') assert.equal(model.closes, 1);
    });
  }
}

for (const restart of [false, true]) {
  for (const outcome of ['resolve', 'reject']) {
    test(`obsolete camera downgrade ${outcome} cannot change a ${restart ? 'new pending downgrade' : 'reset session'}`, async () => {
      const h = harness(), delayed = deferred();
      const { camera } = await start(h);
      camera.track.applyConstraints = () => delayed.promise;
      const oldRequest = h.downgradeCameraQuality();
      assert.equal(h.state().cameraConstraintPending, true);
      h.resetExperience();
      const current = restart ? await start(h) : null;
      const newDelay = deferred();
      if (current) current.camera.track.applyConstraints = () => newDelay.promise;
      const newRequest = current ? h.downgradeCameraQuality() : null;
      const ui = cameraUi(h);
      if (outcome === 'resolve') delayed.resolve();
      else delayed.reject(new Error('obsolete camera constraint failure'));
      await oldRequest;
      assert.deepEqual(cameraUi(h), ui, 'old completion must not change diagnostics, hints, or a newer pending flag');
      assert.equal(h.warnings.length, 0, 'obsolete failures are not current session failures');
      assert.equal(h.video.srcObject, current?.camera ?? null);
      if (current) {
        assert.equal(current.camera.track.stops, 0);
        newDelay.resolve(); await newRequest;
        assert.equal(h.state().cameraQualityIndex, 2);
        assert.equal(h.state().cameraConstraintPending, false);
        assert.equal(h.video.dataset.cameraTier, 'efficient');
      }
    });
  }
}

test('current camera downgrade failure clears the pending flag and allows retry', async () => {
  const h = harness(), delayed = deferred();
  const { camera } = await start(h);
  camera.track.applyConstraints = () => delayed.promise;
  const before = cameraUi(h), pending = h.downgradeCameraQuality();
  delayed.reject(new Error('current camera constraint failure')); await pending;
  assert.deepEqual(cameraUi(h), before);
  assert.equal(h.warnings.length, 1);
  camera.track.applyConstraints = async value => { camera.track.constraints.push(value); };
  await h.downgradeCameraQuality();
  assert.equal(camera.track.constraints.length, 1);
  assert.equal(h.state().cameraQualityIndex, 2);
  assert.equal(h.state().cameraConstraintPending, false);
});

test('successful start, audio toggle, and reset release only acquired resources', async () => {
  const h = harness();
  const { camera, model } = await start(h);
  assert.equal(h.video.srcObject, camera);
  assert.equal(h.sandbox.audioButton.disabled, false);
  const microphone = mediaStream(); h.microphones.push(microphone);
  await h.enableAudioAssist();
  assert.equal(h.state().audioEnabled, true);
  assert.equal(h.contexts[0].analyser.fftSize, 512);
  assert.equal(h.sandbox.audioButton.attributes['aria-pressed'], 'true');
  await h.enableAudioAssist();
  assert.equal(h.state().audioEnabled, false);
  assert.equal(microphone.track.stops, 1);
  assert.equal(h.contexts[0].closes, 1);
  assert.equal(h.contexts[0].analyser.disconnects, 1);
  assert.equal(camera.track.stops, 0);
  h.resetExperience();
  assert.equal(camera.track.stops, 1);
  assert.equal(model.closes, 1);
  assert.equal(h.video.srcObject, null);
  assert.equal(h.state().stream, null);
  assert.equal(h.state().uiState, 'idle');
  assert.equal(h.sandbox.startButton.disabled, false);
  assert.equal(h.sandbox.audioButton.disabled, true);
});

test('model failure after camera acquisition releases the camera and permits retry', async () => {
  const h = harness(), failedModel = deferred(), camera = mediaStream();
  h.cameras.push(camera); h.models.push(failedModel.promise);
  const pending = h.startExperience();
  await flush();
  assert.equal(h.video.srcObject, camera);
  failedModel.reject(new Error('model unavailable'));
  await pending;
  assert.equal(h.state().uiState, 'error');
  assert.equal(camera.track.readyState, 'ended');
  assert.equal(h.video.srcObject, null);
  assert.equal(h.video.pauses, 1);
  assert.equal(h.state().stream, null);
  assert.equal(h.sandbox.startButton.disabled, false);
  const retry = await start(h);
  assert.equal(h.video.srcObject, retry.camera);
  assert.equal(retry.camera.track.stops, 0);
});

test('late camera and model completion after reset cannot revive the experience', async () => {
  const h = harness(), cameraRequest = deferred(), modelRequest = deferred();
  h.cameras.push(cameraRequest.promise); h.models.push(modelRequest.promise);
  const pending = h.startExperience();
  h.resetExperience();
  const camera = mediaStream(), model = faceModel();
  cameraRequest.resolve(camera); modelRequest.resolve(model);
  await pending; await flush();
  assert.equal(h.state().uiState, 'idle');
  assert.equal(camera.track.readyState, 'ended');
  assert.equal(model.closes, 1);
  assert.equal(h.video.srcObject, null);
  assert.equal(h.state().faceLandmarker, null);
});

test('obsolete model rejection does not erase a newer model request or its UI', async () => {
  const h = harness(), oldModel = deferred(), newModel = deferred();
  h.cameras.push(mediaStream()); h.models.push(oldModel.promise);
  const oldStart = h.startExperience(); await flush();
  h.resetExperience();
  const camera = mediaStream(); h.cameras.push(camera); h.models.push(newModel.promise);
  const newStart = h.startExperience(); await flush();
  const cached = h.state().faceLandmarkerLoadPromise;
  assert.ok(cached);
  oldModel.reject(new Error('obsolete failure')); await oldStart;
  assert.equal(h.state().faceLandmarkerLoadPromise, cached);
  assert.equal(h.state().uiState, 'loading');
  assert.equal(h.video.srcObject, camera);
  newModel.resolve(faceModel()); await newStart;
  assert.equal(h.state().uiState, 'running');
});

for (const waitPoint of ['permission', 'resume']) {
  for (const outcome of ['resolve', 'reject']) {
    test(`audio ${waitPoint}: reset plus new session ignores obsolete ${outcome}`, async () => {
      const h = harness(); await start(h);
      const delayed = deferred(), oldMicrophone = mediaStream();
      if (waitPoint === 'permission') h.microphones.push(delayed.promise);
      else { h.microphones.push(oldMicrophone); h.audioPlans.push({ resume: delayed.promise }); }
      const oldRequest = h.enableAudioAssist(); await flush();
      h.resetExperience();
      if (waitPoint === 'resume') {
        assert.equal(oldMicrophone.track.readyState, 'ended');
        assert.equal(h.contexts[0].state, 'closed');
      }
      await start(h);
      const newMicrophone = mediaStream(); h.microphones.push(newMicrophone);
      await h.enableAudioAssist();
      const current = h.state(), ui = audioUi(h);
      if (outcome === 'resolve') delayed.resolve(waitPoint === 'permission' ? oldMicrophone : undefined);
      else delayed.reject(new DOMException('obsolete denial', 'NotAllowedError'));
      await oldRequest;
      assert.equal(h.state().uiState, 'running');
      assert.equal(h.state().audioStream, newMicrophone);
      assert.equal(h.state().audioContext, current.audioContext);
      assert.equal(h.state().audioAnalyser, current.audioAnalyser);
      assert.equal(h.state().audioEnabled, true);
      assert.deepEqual(audioUi(h), ui);
      assert.equal(newMicrophone.track.stops, 0);
      assert.equal(current.audioContext.closes, 0);
      assert.equal(current.audioContext.sources, 1, 'stale continuation must not attach another source');
      if (waitPoint === 'resume') assert.equal(h.contexts[0].sources, 0);
      if (outcome === 'resolve' || waitPoint === 'resume') assert.equal(oldMicrophone.track.readyState, 'ended');
    });
  }
}

for (const waitPoint of ['permission', 'resume']) {
  test(`audio ${waitPoint}: late success leaves a reset session idle`, async () => {
    const h = harness(); await start(h);
    const delayed = deferred(), microphone = mediaStream();
    if (waitPoint === 'permission') h.microphones.push(delayed.promise);
    else { h.microphones.push(microphone); h.audioPlans.push({ resume: delayed.promise }); }
    const pending = h.enableAudioAssist(); await flush();
    h.resetExperience(); const ui = audioUi(h);
    delayed.resolve(waitPoint === 'permission' ? microphone : undefined); await pending;
    assert.equal(microphone.track.readyState, 'ended');
    assert.equal(h.state().audioEnabled, false);
    assert.equal(h.state().audioStream, null);
    assert.equal(h.state().audioContext, null);
    assert.equal(h.state().uiState, 'idle');
    assert.deepEqual(audioUi(h), ui);
    assert.ok(h.contexts.every(context => context.state === 'closed'));
  });
}

test('current microphone failure leaves camera running and audio retry usable', async () => {
  const h = harness(); const { camera } = await start(h);
  h.microphones.push(Promise.reject(new DOMException('denied', 'NotAllowedError')));
  await h.enableAudioAssist();
  assert.equal(h.state().uiState, 'running');
  assert.equal(h.state().audioEnabled, false);
  assert.equal(h.sandbox.audioButton.disabled, false);
  assert.equal(camera.track.stops, 0);
  h.microphones.push(mediaStream()); await h.enableAudioAssist();
  assert.equal(h.state().audioEnabled, true);
});

test('duplicate audio clicks during permission request acquire only one stream', async () => {
  const h = harness(); await start(h);
  const delayed = deferred(), microphone = mediaStream(); h.microphones.push(delayed.promise);
  const pending = h.enableAudioAssist();
  await h.enableAudioAssist();
  delayed.resolve(microphone); await pending;
  assert.equal(h.mediaRequests.filter(request => request.audio).length, 1);
  assert.equal(h.state().audioStream, microphone);
  assert.equal(h.contexts.length, 1);
  assert.equal(h.state().audioEnabled, true);
});

test('current audio resume failure closes resources and permits another attempt', async () => {
  const h = harness(); const { camera } = await start(h);
  const delayed = deferred(), microphone = mediaStream();
  h.microphones.push(microphone); h.audioPlans.push({ resume: delayed.promise });
  const pending = h.enableAudioAssist(); await flush();
  delayed.reject(new Error('resume failed')); await pending;
  assert.equal(microphone.track.readyState, 'ended');
  assert.equal(h.contexts[0].state, 'closed');
  assert.equal(h.state().audioStream, null);
  assert.equal(h.state().audioContext, null);
  assert.equal(h.sandbox.audioButton.disabled, false);
  assert.equal(camera.track.stops, 0);
  h.microphones.push(mediaStream()); await h.enableAudioAssist();
  assert.equal(h.state().audioEnabled, true);
});

for (const refreshRate of [30, 50, 60, 75, 90, 120, 144]) {
  test(`${refreshRate} Hz rAF keeps the expected render rate without drift`, () => {
    const h = harness(), duration = 20, expected = Math.min(refreshRate, 60);
    for (let index = 0; index <= refreshRate * duration; index += 1) h.tick(index * 1000 / refreshRate);
    assert.ok(Math.abs(h.frames.length - (expected * duration + 1)) <= 2,
      `rendered ${h.frames.length} frames; expected about ${expected * duration + 1}`);
    assert.ok(Math.abs(h.state().renderFps - expected) <= 2);
    assert.equal(h.scheduled.length, refreshRate * duration + 1, 'exactly one next rAF per callback');
  });
}

test('long render stall advances real elapsed time once without catching up frames', () => {
  const h = harness();
  h.tick(0); h.tick(1000 / 60);
  const before = h.frames.length;
  h.tick(30_000);
  assert.equal(h.frames.length, before + 1);
  assert.equal(h.frames.at(-1).delta, 0.05);
  assert.ok(h.frames.at(-1).elapsed > 29);
  h.tick(30_000);
  assert.equal(h.frames.length, before + 1, 'same timestamp must not catch up missed frames');
  for (let index = 1; index <= 144; index += 1) h.tick(30_000 + index * 1000 / 144);
  assert.ok(Math.abs(h.frames.length - (before + 61)) <= 1);
});

for (const refreshRate of [30, 75, 90, 144]) {
  test(`${refreshRate} Hz retains camera performance downgrade behavior`, async () => {
    const h = harness(); const { camera } = await start(h);
    for (let index = 0; index <= refreshRate * 13; index += 1) h.tick(index * 1000 / refreshRate);
    await flush();
    assert.equal(camera.track.constraints.length, refreshRate === 30 ? 1 : 0);
    assert.equal(h.state().cameraQualityIndex, refreshRate === 30 ? 2 : 1);
  });
}
