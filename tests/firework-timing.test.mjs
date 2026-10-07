import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

// Use the project's existing compiler; no browser, camera, or new dependency.
const source = readFileSync(new URL('../src/procedural-firework.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 },
}).outputText;
const { ProceduralFireworkSystem } = await import(
  `data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`
);

const viewport = { width: 800, height: 600 };
const head = { cx: 400, cy: 320, rx: 70, ry: 100, angle: 0 };

function withSeed(seed, run) {
  const originalRandom = Math.random;
  let state = seed >>> 0;
  Math.random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  try { return run(); } finally { Math.random = originalRandom; }
}

// Match render's ordering: age existing effects, spawn at most one new group,
// then perform the original bounded motion step. Inspect private fields only
// in tests to catch non-finite physics and incorrect flash/cooldown ageing.
function frame(system, elapsed, bounds = null, spawn = () => {}) {
  const dt = Math.min(0.05, elapsed);
  if (elapsed > dt) system.elapseStalledTime(elapsed - dt);
  spawn();
  system.update(dt, bounds, elapsed);
}

for (const fps of [60, 30, 20, 15, 10, 5]) {
  test(`${fps} FPS: effects expire in real time after emission stops`, () => withSeed(7, () => {
    const system = new ProceduralFireworkSystem();
    system.spawn(null, viewport, 'entry');
    let elapsed = 0;
    while (system.activeCount && elapsed < 10) {
      frame(system, 1 / fps);
      elapsed += 1 / fps;
    }
    assert.equal(system.activeCount, 0);
    assert.equal(system.flashes.length, 0);
    assert.ok(elapsed <= 1.61 + 1 / fps, `clearance took ${elapsed}s`);
  }));
}

for (const gap of [2, 30]) {
  test(`${gap}s stall: retire old effects without ageing the new group through the stall`, () => withSeed(11, () => {
    const system = new ProceduralFireworkSystem();
    system.spawn(head, viewport, 'entry');
    frame(system, 1 / 60, head);
    const oldSparks = new Set(system.sparks);
    let newbornAges;
    const movedHead = { ...head, cx: head.cx + 120 };
    frame(system, gap, movedHead, () => {
      assert.equal(system.activeCount, 0);
      assert.equal(system.flashes.length, 0);
      system.spawn(movedHead, viewport, 'sustain');
      newbornAges = system.sparks.map(spark => spark.age);
    });
    assert.ok(system.activeCount >= 120 && system.activeCount <= 125);
    system.sparks.forEach((spark, index) => {
      assert.ok(!oldSparks.has(spark));
      assert.equal(spark.age, newbornAges[index] + 0.05);
    });
    assert.equal(system.flashes.length, 2);
    assert.ok(system.flashes.every(flash => flash.age === 0.05));
    assert.equal(system.headVelocityX, 0);
    assert.equal(system.headVelocityY, 0);
    assert.equal(system.activeCollisionCount, 0);
  }));
}

test('stall ageing does not simulate motion and advances collision cooldown', () => withSeed(3, () => {
  const system = new ProceduralFireworkSystem();
  system.spawn(null, viewport, 'sustain');
  const spark = system.sparks[0];
  const position = [spark.x, spark.y, spark.vx, spark.vy];
  spark.collisionCooldown = 0.14;
  const age = spark.age;
  system.elapseStalledTime(0.1);
  assert.deepEqual([spark.x, spark.y, spark.vx, spark.vy], position);
  assert.ok(Math.abs(spark.collisionCooldown - 0.04) < 1e-10);
  assert.equal(spark.age, age + 0.1);
}));

test('head velocity uses the real sample interval, not the bounded motion interval', () => {
  const system = new ProceduralFireworkSystem();
  frame(system, 0.1, head);
  frame(system, 0.1, { ...head, cx: head.cx + 10 });
  assert.ok(Math.abs(system.headVelocityX - 76) < 1e-10);
});

test('continuous paired bursts stay bounded with finite physics under slow frames', () => withSeed(19, () => {
  for (const fps of [60, 10, 5]) {
    const system = new ProceduralFireworkSystem();
    let lastSpawn = -1.3;
    for (let index = 0; index < fps * 30; index += 1) {
      const now = index / fps;
      const bounds = { ...head, cx: head.cx + Math.sin(now * 5) * 160 };
      frame(system, 1 / fps, bounds, () => {
        if (now - lastSpawn >= 1.3 - 1e-9) {
          system.spawn(bounds, viewport, index === 0 ? 'entry' : 'sustain');
          lastSpawn = now;
        }
      });
      assert.ok(system.activeCount <= 220);
      assert.ok(system.flashes.length <= 6);
      for (const spark of system.sparks) {
        for (const field of ['x', 'y', 'vx', 'vy', 'age', 'life']) {
          assert.ok(Number.isFinite(spark[field]), `${field} is not finite at ${fps} FPS`);
        }
      }
    }
    assert.ok(system.activeCollisionCount > 0, `moving-head fixture must exercise collisions at ${fps} FPS`);
    // Stop moving and emitting: even collision-extended lives must terminate.
    for (let index = 0; index < fps * 4; index += 1) frame(system, 1 / fps);
    assert.equal(system.activeCount, 0);
    assert.equal(system.flashes.length, 0);
  }
}));

test('integration keeps stall ageing before every render-time spawn and does not catch up bursts', () => {
  const main = readFileSync(new URL('../src/main.ts', import.meta.url), 'utf8');
  const render = main.slice(main.indexOf('function render(now:'), main.indexOf('async function createLandmarker'));
  const ageIndex = render.indexOf('proceduralFireworks.elapseStalledTime(');
  assert.ok(ageIndex >= 0);
  assert.ok(ageIndex < render.indexOf("spawnProceduralFirework('entry')"));
  assert.ok(ageIndex < render.indexOf('updateInteraction(performance.now())'));
  assert.ok(ageIndex < render.indexOf('updateSustainedFireworks(now)'));
  assert.ok(render.includes('proceduralFireworks.update(deltaSeconds, latestFeatures.headBounds, elapsedSeconds)'));
  const sustain = main.slice(main.indexOf('function updateSustainedFireworks('), main.indexOf('function updateParticles('));
  assert.equal(sustain.match(/spawnProceduralFirework\('sustain'\)/g)?.length, 1);
  assert.ok(!/\b(while|for)\s*\(/.test(sustain));
});
