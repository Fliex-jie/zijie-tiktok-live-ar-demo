export interface FireworkHeadBounds {
  cx: number;
  cy: number;
  rx: number;
  ry: number;
  angle: number;
}

export interface FireworkViewport {
  width: number;
  height: number;
}

type Spark = {
  x: number;
  y: number;
  previousX: number;
  previousY: number;
  vx: number;
  vy: number;
  radius: number;
  trail: number;
  age: number;
  life: number;
  gravity: number;
  drag: number;
  curveBias: number;
  color: string;
  glow: number;
  detailPhase: number;
  collisionCooldown: number;
  collisionCount: number;
  touchingHead: boolean;
  interactive: boolean;
};

type BurstFlash = {
  x: number;
  y: number;
  age: number;
  life: number;
  radius: number;
  color: string;
  kind: 'burst' | 'impact';
  nx: number;
  ny: number;
  strength: number;
};

type Collision = {
  x: number;
  y: number;
  nx: number;
  ny: number;
  tailOnly: boolean;
};

type TrailGeometry = {
  tailX: number;
  tailY: number;
  controlX: number;
  controlY: number;
  headX: number;
  headY: number;
};

type BurstTheme = {
  pearls: readonly string[];
  golds: readonly string[];
};

// A burst uses one coherent pearl theme instead of sampling the entire palette
// independently per spark. Alternating a cool lavender and a pink-champagne
// family keeps successive bursts varied while preserving one clear identity.
const BURST_THEMES: readonly BurstTheme[] = [
  {
    pearls: ['#ffffff', '#e8deff', '#cab2ff', '#aa8cff'],
    golds: ['#fff1a8', '#ffd15a'],
  },
  {
    pearls: ['#ffffff', '#ffd9f1', '#ff8fd4', '#d8baff'],
    golds: ['#fff0a0', '#ffc95c'],
  },
];
const MAX_SPARKS = 220;
const MAX_FLASHES = 6;
const TRAIL_TIME_WINDOW = 0.132;
const TRAIL_CURVE_GRAVITY_GAIN = 5.2;
const TRAIL_DRAW_SEGMENTS = 12;
const TRAIL_COLLISION_SEGMENTS = TRAIL_DRAW_SEGMENTS;

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function quadraticPoint(
  geometry: TrailGeometry,
  progress: number,
): { x: number; y: number } {
  const inverse = 1 - progress;
  return {
    x: inverse * inverse * geometry.tailX
      + 2 * inverse * progress * geometry.controlX
      + progress * progress * geometry.headX,
    y: inverse * inverse * geometry.tailY
      + 2 * inverse * progress * geometry.controlY
      + progress * progress * geometry.headY,
  };
}

function sparkTrailGeometry(spark: Spark): TrailGeometry {
  const speed = Math.max(1, Math.hypot(spark.vx, spark.vy));
  const trailLength = Math.min(spark.trail, speed * TRAIL_TIME_WINDOW);
  const travelTime = trailLength / speed;
  const ux = spark.vx / speed;
  const uy = spark.vy / speed;
  const nx = -uy;
  const ny = ux;
  const gravityCurve = 0.5
    * spark.gravity
    * travelTime
    * travelTime
    * TRAIL_CURVE_GRAVITY_GAIN;

  return {
    tailX: spark.x - ux * trailLength,
    tailY: spark.y - uy * trailLength + gravityCurve,
    // The control point preserves the current velocity at the bright head;
    // a small per-particle bias avoids a mechanically identical set of arcs.
    controlX: spark.x - ux * trailLength * 0.5 + nx * trailLength * spark.curveBias,
    controlY: spark.y - uy * trailLength * 0.5
      + ny * trailLength * spark.curveBias
      + gravityCurve * 0.18,
    headX: spark.x,
    headY: spark.y,
  };
}

function toHeadLocal(bounds: FireworkHeadBounds, x: number, y: number): { x: number; y: number } {
  const cosine = Math.cos(bounds.angle);
  const sine = Math.sin(bounds.angle);
  const dx = x - bounds.cx;
  const dy = y - bounds.cy;
  return {
    x: dx * cosine + dy * sine,
    y: -dx * sine + dy * cosine,
  };
}

function toStage(bounds: FireworkHeadBounds, x: number, y: number): { x: number; y: number } {
  const cosine = Math.cos(bounds.angle);
  const sine = Math.sin(bounds.angle);
  return {
    x: bounds.cx + x * cosine - y * sine,
    y: bounds.cy + x * sine + y * cosine,
  };
}

function findEllipseCollision(spark: Spark, bounds: FireworkHeadBounds): Collision | null {
  const previous = toHeadLocal(bounds, spark.previousX, spark.previousY);
  const current = toHeadLocal(bounds, spark.x, spark.y);
  const ellipseValue = (point: { x: number; y: number }): number =>
    (point.x * point.x) / (bounds.rx * bounds.rx) + (point.y * point.y) / (bounds.ry * bounds.ry);
  const previousValue = ellipseValue(previous);
  const currentValue = ellipseValue(current);

  let contact: { x: number; y: number };
  let tailOnly = false;
  if (currentValue < 1) {
    if (previousValue >= 1) {
      let outside = 0;
      let inside = 1;
      for (let iteration = 0; iteration < 7; iteration += 1) {
        const midpoint = (outside + inside) / 2;
        const point = {
          x: previous.x + (current.x - previous.x) * midpoint,
          y: previous.y + (current.y - previous.y) * midpoint,
        };
        if (ellipseValue(point) >= 1) outside = midpoint;
        else inside = midpoint;
      }
      contact = {
        x: previous.x + (current.x - previous.x) * inside,
        y: previous.y + (current.y - previous.y) * inside,
      };
    } else {
      // The head moved onto the spark between face samples. Project the spark
      // back to the moving ellipse boundary so head motion also has agency.
      const projection = 1 / Math.sqrt(Math.max(0.0001, currentValue));
      contact = { x: current.x * projection, y: current.y * projection };
    }
  } else {
    // The visible trail can touch the head even when the bright leading point
    // is already outside it. Sample the same quadratic curve used for drawing
    // so visual contact and physical contact stay in sync.
    const geometry = sparkTrailGeometry(spark);
    const collisionRx = bounds.rx + spark.radius;
    const collisionRy = bounds.ry + spark.radius;
    let trailContact: { x: number; y: number } | null = null;
    let segmentEnd = toHeadLocal(bounds, geometry.headX, geometry.headY);
    // Walk from the bright head toward the tail so the first resolved contact
    // is the visually leading one. Testing the closest point on each segment
    // prevents a thin arc from slipping between two discrete samples.
    for (let step = TRAIL_COLLISION_SEGMENTS - 1; step >= 0; step -= 1) {
      const stageStart = quadraticPoint(geometry, step / TRAIL_COLLISION_SEGMENTS);
      const segmentStart = toHeadLocal(bounds, stageStart.x, stageStart.y);
      const startX = segmentEnd.x / collisionRx;
      const startY = segmentEnd.y / collisionRy;
      const deltaX = segmentStart.x / collisionRx - startX;
      const deltaY = segmentStart.y / collisionRy - startY;
      const lengthSquared = Math.max(0.0001, deltaX * deltaX + deltaY * deltaY);
      const closestProgress = clamp(
        -(startX * deltaX + startY * deltaY) / lengthSquared,
        0,
        1,
      );
      const closestX = startX + deltaX * closestProgress;
      const closestY = startY + deltaY * closestProgress;
      if (closestX * closestX + closestY * closestY <= 1) {
        trailContact = {
          x: segmentEnd.x + (segmentStart.x - segmentEnd.x) * closestProgress,
          y: segmentEnd.y + (segmentStart.y - segmentEnd.y) * closestProgress,
        };
        break;
      }
      segmentEnd = segmentStart;
    }
    if (!trailContact) return null;
    const projection = 1 / Math.sqrt(Math.max(0.0001, ellipseValue(trailContact)));
    contact = { x: trailContact.x * projection, y: trailContact.y * projection };
    tailOnly = true;
  }

  const gradientX = contact.x / (bounds.rx * bounds.rx);
  const gradientY = contact.y / (bounds.ry * bounds.ry);
  const gradientLength = Math.max(0.0001, Math.hypot(gradientX, gradientY));
  const localNx = gradientX / gradientLength;
  const localNy = gradientY / gradientLength;
  const cosine = Math.cos(bounds.angle);
  const sine = Math.sin(bounds.angle);
  const stageContact = toStage(bounds, contact.x, contact.y);
  return {
    x: stageContact.x,
    y: stageContact.y,
    nx: localNx * cosine - localNy * sine,
    ny: localNx * sine + localNy * cosine,
    tailOnly,
  };
}

function chooseColor(index: number, goldShare: number, theme: BurstTheme): string {
  if ((index * 17) % 100 < goldShare * 100) return theme.golds[index % theme.golds.length];
  return theme.pearls[index % theme.pearls.length];
}

function withAlpha(hex: string, alpha: number): string {
  const value = Number.parseInt(hex.slice(1), 16);
  const red = (value >> 16) & 255;
  const green = (value >> 8) & 255;
  const blue = value & 255;
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function traceTaperedSpark(
  target: CanvasRenderingContext2D,
  geometry: TrailGeometry,
  radius: number,
): void {
  const halfWidth = radius * 0.66;

  target.beginPath();
  target.moveTo(geometry.tailX, geometry.tailY);

  // Build a tapered ribbon around a quadratic centre line. Both sides reuse
  // the same curve, so the streak bends under gravity without looking like a
  // rigid needle or changing its visual weight along the arc.
  for (let step = 1; step <= TRAIL_DRAW_SEGMENTS; step += 1) {
    const progress = step / TRAIL_DRAW_SEGMENTS;
    const inverse = 1 - progress;
    const x = inverse * inverse * geometry.tailX
      + 2 * inverse * progress * geometry.controlX
      + progress * progress * geometry.headX;
    const y = inverse * inverse * geometry.tailY
      + 2 * inverse * progress * geometry.controlY
      + progress * progress * geometry.headY;
    const tangentX = 2 * inverse * (geometry.controlX - geometry.tailX)
      + 2 * progress * (geometry.headX - geometry.controlX);
    const tangentY = 2 * inverse * (geometry.controlY - geometry.tailY)
      + 2 * progress * (geometry.headY - geometry.controlY);
    const tangentLength = Math.max(0.001, Math.hypot(tangentX, tangentY));
    const width = halfWidth * Math.pow(progress, 0.72);
    target.lineTo(
      x - (tangentY / tangentLength) * width,
      y + (tangentX / tangentLength) * width,
    );
  }

  const headDirection = Math.atan2(
    geometry.headY - geometry.controlY,
    geometry.headX - geometry.controlX,
  );
  target.arc(
    geometry.headX,
    geometry.headY,
    halfWidth,
    headDirection + Math.PI / 2,
    headDirection - Math.PI / 2,
    true,
  );

  for (let step = TRAIL_DRAW_SEGMENTS - 1; step >= 0; step -= 1) {
    const progress = step / TRAIL_DRAW_SEGMENTS;
    const inverse = 1 - progress;
    const x = inverse * inverse * geometry.tailX
      + 2 * inverse * progress * geometry.controlX
      + progress * progress * geometry.headX;
    const y = inverse * inverse * geometry.tailY
      + 2 * inverse * progress * geometry.controlY
      + progress * progress * geometry.headY;
    const tangentX = 2 * inverse * (geometry.controlX - geometry.tailX)
      + 2 * progress * (geometry.headX - geometry.controlX);
    const tangentY = 2 * inverse * (geometry.controlY - geometry.tailY)
      + 2 * progress * (geometry.headY - geometry.controlY);
    const tangentLength = Math.max(0.001, Math.hypot(tangentX, tangentY));
    const width = halfWidth * Math.pow(progress, 0.72);
    target.lineTo(
      x + (tangentY / tangentLength) * width,
      y - (tangentX / tangentLength) * width,
    );
  }
  target.closePath();
}

function drawDerivedAfterglow(
  target: CanvasRenderingContext2D,
  geometry: TrailGeometry,
  radius: number,
  detailPhase: number,
  alpha: number,
): void {
  // Only a little over half the streaks receive two micro-sparks. These are
  // derived from the visible curve rather than simulated particles, so the
  // richer trail adds no update state, collision tests, or per-frame objects.
  if (detailPhase > Math.PI * 1.12) return;
  const tangentX = geometry.controlX - geometry.tailX;
  const tangentY = geometry.controlY - geometry.tailY;
  const tangentLength = Math.max(0.001, Math.hypot(tangentX, tangentY));
  const ux = tangentX / tangentLength;
  const uy = tangentY / tangentLength;
  const nx = -uy;
  const ny = ux;

  for (let step = 0; step < 2; step += 1) {
    const distance = radius * (2.0 + step * 2.7);
    const lateralOffset = Math.sin(detailPhase + step * 2.35)
      * radius
      * (0.26 + step * 0.16);
    target.globalAlpha = alpha * (0.36 - step * 0.12);
    target.beginPath();
    target.arc(
      geometry.tailX - ux * distance + nx * lateralOffset,
      geometry.tailY - uy * distance + ny * lateralOffset,
      Math.max(0.64, radius * (0.28 - step * 0.045)),
      0,
      Math.PI * 2,
    );
    target.fill();
  }
}

export class ProceduralFireworkSystem {
  private sparks: Spark[] = [];
  private flashes: BurstFlash[] = [];
  private sequence = 0;
  private activeSide: -1 | 1 = -1;
  private previousHeadCenter: { x: number; y: number } | null = null;
  private previousCollisionBounds: FireworkHeadBounds | null = null;
  private headSampleAge = 0;
  private headVelocityX = 0;
  private headVelocityY = 0;
  collisionCount = 0;
  activeCollisionCount = 0;

  get activeCount(): number {
    return this.sparks.length;
  }

  clear(): void {
    this.sparks.length = 0;
    this.flashes.length = 0;
    this.previousHeadCenter = null;
    this.previousCollisionBounds = null;
    this.headSampleAge = 0;
    this.headVelocityX = 0;
    this.headVelocityY = 0;
    this.activeSide = -1;
    this.collisionCount = 0;
    this.activeCollisionCount = 0;
  }

  spawn(bounds: FireworkHeadBounds | null, viewport: FireworkViewport, mode: 'entry' | 'sustain'): void {
    const previousSparkCount = this.sparks.length;
    const side = this.sequence % 2 === 0 ? -1 : 1;
    const isEntry = mode === 'entry';
    if (isEntry) {
      if (bounds) {
        const leftSpace = bounds.cx - bounds.rx;
        const rightSpace = viewport.width - bounds.cx - bounds.rx;
        const meaningfulDifference = viewport.width * 0.06;
        this.activeSide = Math.abs(rightSpace - leftSpace) >= meaningfulDifference
          ? rightSpace > leftSpace ? 1 : -1
          : this.activeSide === -1 ? 1 : -1;
      } else {
        this.activeSide = this.activeSide === -1 ? 1 : -1;
      }
    }
    this.sequence += 1;
    const horizontalMargin = Math.min(viewport.width * 0.22, 164);
    const verticalMargin = Math.min(viewport.height * 0.22, 126);
    const mainX = bounds
      ? clamp(bounds.cx + side * bounds.rx * 1.42, horizontalMargin, viewport.width - horizontalMargin)
      : viewport.width * (side < 0 ? 0.38 : 0.62);
    const mainY = bounds
      ? clamp(bounds.cy - bounds.ry * 1.34, verticalMargin, viewport.height * 0.34)
      : viewport.height * 0.24;

    const emitBurst = (
      originX: number,
      originY: number,
      spokeCount: number,
      scale: number,
      goldShare: number,
      colorOffset: number,
      interactive = true,
    ): void => {
      const rotationOffset = Math.random() * Math.PI * 2;
      const angularStep = Math.PI * 2 / spokeCount;
      const theme = BURST_THEMES[
        (this.sequence + (colorOffset > 0 ? 1 : 0)) % BURST_THEMES.length
      ];
      let colorIndex = 0;

      const addSpark = (
        angle: number,
        speed: number,
        radius: number,
        trail: number,
        life: number,
        gravity: number,
        drag: number,
        startDelay: number,
        core: boolean,
      ): void => {
        const color = core
          ? '#ffffff'
          : chooseColor(
            colorIndex + this.sequence * 7 + colorOffset,
            goldShare,
            theme,
          );
        colorIndex += 1;
        this.sparks.push({
          x: originX,
          y: originY,
          previousX: originX,
          previousY: originY,
          vx: Math.cos(angle) * speed,
          vy: Math.sin(angle) * speed - 12 * scale,
          radius,
          trail,
          age: -Math.random() * startDelay,
          life,
          gravity,
          drag,
          curveBias: (Math.random() - 0.5) * 0.075,
          color,
          // Glow belongs to the white core only. The colored outer particles
          // keep a clean single body instead of regaining the rejected shell.
          glow: core ? (7 + Math.random() * 5) * Math.sqrt(scale) : 0,
          detailPhase: Math.random() * Math.PI * 2,
          collisionCooldown: 0,
          collisionCount: 0,
          touchingHead: false,
          interactive,
        });
      };

      for (let index = 0; index < spokeCount; index += 1) {
        // One evenly-spaced outer spark per sector gives the explosion a clear,
        // continuous silhouette. The jitter stays well inside a sector so the
        // body never collapses into random clumps or two disconnected rings.
        const baseAngle = rotationOffset + index * angularStep;
        const outerAngle = baseAngle + (Math.random() - 0.5) * angularStep * 0.24;
        const silhouetteScale = 0.96
          + Math.sin(index * 2.399963 + rotationOffset) * 0.055;
        // Slower launch speed plus stronger drag keeps the whole crown inside
        // a readable radius instead of letting it disperse across the stage.
        const outerSpeed = (178 + Math.random() * 28) * scale * silhouetteScale;
        addSpark(
          outerAngle,
          outerSpeed,
          (2.00 + Math.random() * 0.72) * Math.sqrt(scale),
          (36 + Math.random() * 12) * Math.sqrt(scale),
          1.48 + Math.random() * 0.10,
          (52 + Math.random() * 18) * scale,
          1.88 + Math.random() * 0.14,
          0.026,
          false,
        );

        // A companion on four out of five spokes sits close to the same ray.
        // Its broad speed range fills the body instead of drawing a second ring.
        // This builds a cohesive radial
        // mass like the reference without copying its very high particle count.
        if ((index + this.sequence) % 5 !== 0) {
          const direction = index % 2 === 0 ? -1 : 1;
          const companionAngle = baseAngle
            + direction * angularStep * (0.20 + Math.random() * 0.13);
          addSpark(
            companionAngle,
            outerSpeed * (0.68 + Math.random() * 0.22),
            (1.68 + Math.random() * 0.58) * Math.sqrt(scale),
            (29 + Math.random() * 10) * Math.sqrt(scale),
            1.45 + Math.random() * 0.10,
            (58 + Math.random() * 18) * scale,
            1.96 + Math.random() * 0.14,
            0.032,
            false,
          );
        }

        // Inner rays cover alternate sectors with deliberately varied speeds,
        // connecting the flash core to the crown without forming a third ring.
        if ((index + colorOffset) % 3 !== 0) {
          const innerAngle = baseAngle + (Math.random() - 0.5) * angularStep * 0.26;
          addSpark(
            innerAngle,
            outerSpeed * (0.25 + Math.random() * 0.30),
            (1.42 + Math.random() * 0.48) * Math.sqrt(scale),
            (26 + Math.random() * 11) * Math.sqrt(scale),
            1.42 + Math.random() * 0.10,
            (64 + Math.random() * 16) * scale,
            2.04 + Math.random() * 0.14,
            0.020,
            true,
          );
        }
      }
      this.flashes.push({
        x: originX,
        y: originY,
        age: 0,
        life: 0.50,
        radius: (isEntry ? 32 : 25) * scale,
        color: '#ffffff',
        kind: 'burst',
        nx: 0,
        ny: -1,
        strength: 1,
      });
    };

    // Keep the approved main burst intact. Sustained laughter also gets a
    // smaller companion in the same spawn call, swapping sides as a pair.
    // The companion uses fewer spokes rather than doubling the main burst.
    emitBurst(mainX, mainY, isEntry ? 44 : 36, 1.18, isEntry ? 0.13 : 0.24, 0);
    const accentX = bounds
      ? clamp(bounds.cx - side * bounds.rx * 1.34, horizontalMargin * 0.78, viewport.width - horizontalMargin * 0.78)
      : viewport.width * (side < 0 ? 0.64 : 0.36);
    const accentY = bounds
      ? clamp(bounds.cy - bounds.ry * 0.52, viewport.height * 0.18, viewport.height * 0.52)
      : viewport.height * 0.42;
    emitBurst(accentX, accentY, isEntry ? 22 : 14, isEntry ? 0.78 : 0.68, 0.72, 41);

    if (isEntry) {
      // Add one smaller interactive satellite on the roomier side without
      // replacing either approved burst. Its resting gap means the viewer has
      // to move the head toward it before any physical response is possible.
      const satelliteSide = this.activeSide;
      const satelliteReach = clamp(viewport.height * 0.18, 70, 102);
      const satelliteOffset = bounds
        ? bounds.rx + satelliteReach + Math.max(14, bounds.rx * 0.14)
        : viewport.width * 0.24;
      const satelliteX = bounds
        ? clamp(
          bounds.cx + satelliteSide * satelliteOffset,
          satelliteReach * 0.82,
          viewport.width - satelliteReach * 0.82,
        )
        : viewport.width * (satelliteSide < 0 ? 0.25 : 0.75);
      const satelliteY = bounds
        ? clamp(bounds.cy + bounds.ry * 0.08, viewport.height * 0.24, viewport.height * 0.68)
        : viewport.height * 0.48;
      emitBurst(satelliteX, satelliteY, 16, 0.65, 0.42, 73, true);
    }
    const overflow = this.sparks.length - MAX_SPARKS;
    if (overflow > 0) {
      // Insertion order follows the burst's angles. Removing a contiguous
      // prefix cuts a visible wedge from its crown. Retire the most faded
      // existing sparks instead, keeping the new burst intact. Normalized
      // age follows the fade curve; sorting runs only on overflow, not frames.
      const retiring = new Set(
        this.sparks.slice(0, previousSparkCount)
          .sort((a, b) => b.age / b.life - a.age / a.life)
          .slice(0, overflow),
      );
      let writeIndex = 0;
      for (const spark of this.sparks) {
        if (!retiring.has(spark)) this.sparks[writeIndex++] = spark;
      }
      this.sparks.length = writeIndex;
    }
    while (this.flashes.length > MAX_FLASHES) this.flashes.shift();
  }

  private updateHeadVelocity(deltaSeconds: number, bounds: FireworkHeadBounds | null): void {
    this.headSampleAge += deltaSeconds;
    if (!bounds) {
      this.previousHeadCenter = null;
      this.headVelocityX = 0;
      this.headVelocityY = 0;
      this.headSampleAge = 0;
      return;
    }

    if (!this.previousHeadCenter) {
      this.previousHeadCenter = { x: bounds.cx, y: bounds.cy };
      this.headSampleAge = 0;
      return;
    }

    const deltaX = bounds.cx - this.previousHeadCenter.x;
    const deltaY = bounds.cy - this.previousHeadCenter.y;
    if (Math.hypot(deltaX, deltaY) >= 0.75) {
      // Face landmarks update at a lower cadence than rendering. Measure over
      // the elapsed landmark interval, then clamp tracking noise and camera
      // jumps before using head motion as a physical impulse.
      const sampleDuration = clamp(this.headSampleAge, 0.045, 0.24);
      let velocityX = deltaX / sampleDuration;
      let velocityY = deltaY / sampleDuration;
      const velocityLength = Math.max(0.001, Math.hypot(velocityX, velocityY));
      const velocityScale = Math.min(1, 900 / velocityLength);
      velocityX *= velocityScale;
      velocityY *= velocityScale;
      this.headVelocityX = this.headVelocityX * 0.24 + velocityX * 0.76;
      this.headVelocityY = this.headVelocityY * 0.24 + velocityY * 0.76;
      this.previousHeadCenter = { x: bounds.cx, y: bounds.cy };
      this.headSampleAge = 0;
      return;
    }

    const decay = Math.exp(-2.6 * deltaSeconds);
    this.headVelocityX *= decay;
    this.headVelocityY *= decay;
  }

  private collisionBoundsForFrame(bounds: FireworkHeadBounds | null): FireworkHeadBounds[] {
    if (!bounds) {
      this.previousCollisionBounds = null;
      return [];
    }

    const current = { ...bounds };
    const previous = this.previousCollisionBounds;
    if (!previous) {
      this.previousCollisionBounds = current;
      return [current];
    }

    const deltaX = current.cx - previous.cx;
    const deltaY = current.cy - previous.cy;
    const displacement = Math.hypot(deltaX, deltaY);
    const sizeChange = Math.max(
      Math.abs(current.rx - previous.rx),
      Math.abs(current.ry - previous.ry),
    );
    let angleDelta = current.angle - previous.angle;
    while (angleDelta > Math.PI) angleDelta -= Math.PI * 2;
    while (angleDelta < -Math.PI) angleDelta += Math.PI * 2;
    this.previousCollisionBounds = current;

    if (displacement < 0.5 && sizeChange < 0.5 && Math.abs(angleDelta) < 0.01) {
      return [current];
    }

    // Bridge the 10 Hz face-tracking gaps so a fast head swipe cannot jump
    // completely over a particle between two landmark samples. Very large
    // jumps are treated as tracking reacquisition rather than physical motion.
    // A deliberate head swipe can easily cover two or three face radii in a
    // single landmark interval. Only reject much larger jumps as tracker
    // reacquisition; the previous 2.2x cutoff discarded the strongest hits.
    if (displacement > Math.max(current.rx, current.ry) * 4.0) return [current];
    const sampleSpacing = Math.max(7, Math.min(current.rx, current.ry) * 0.16);
    const sampleCount = clamp(Math.ceil(displacement / sampleSpacing), 2, 6);
    const samples: FireworkHeadBounds[] = [];
    for (let step = 1; step <= sampleCount; step += 1) {
      const progress = step / sampleCount;
      samples.push({
        cx: previous.cx + deltaX * progress,
        cy: previous.cy + deltaY * progress,
        rx: previous.rx + (current.rx - previous.rx) * progress,
        ry: previous.ry + (current.ry - previous.ry) * progress,
        angle: previous.angle + angleDelta * progress,
      });
    }
    return samples;
  }

  // Account for time omitted by the bounded motion step, before this frame
  // spawns anything. Old effects expire on time without ageing a new burst
  // through the stall or replaying expensive physics/collisions to catch up.
  elapseStalledTime(seconds: number): void {
    if (seconds <= 0) return;
    for (let index = this.flashes.length - 1; index >= 0; index -= 1) {
      const flash = this.flashes[index];
      flash.age += seconds;
      if (flash.age >= flash.life) this.flashes.splice(index, 1);
    }
    for (let index = this.sparks.length - 1; index >= 0; index -= 1) {
      const spark = this.sparks[index];
      spark.age += seconds;
      spark.collisionCooldown = Math.max(0, spark.collisionCooldown - seconds);
      if (spark.age >= spark.life) this.sparks.splice(index, 1);
    }
  }

  update(
    deltaSeconds: number,
    bounds: FireworkHeadBounds | null,
    elapsedSeconds = deltaSeconds,
  ): void {
    // A resumed tab has no reliable head-motion path through the missing
    // frames. Reacquire instead of sweeping fresh particles across stale data.
    if (elapsedSeconds > 0.25) {
      this.previousHeadCenter = null;
      this.previousCollisionBounds = null;
      this.headSampleAge = 0;
      this.headVelocityX = 0;
      this.headVelocityY = 0;
    }
    const collisionBounds = this.collisionBoundsForFrame(bounds);
    this.updateHeadVelocity(elapsedSeconds, bounds);
    for (let index = this.flashes.length - 1; index >= 0; index -= 1) {
      const flash = this.flashes[index];
      flash.age += deltaSeconds;
      if (flash.age >= flash.life) this.flashes.splice(index, 1);
    }

    for (let index = this.sparks.length - 1; index >= 0; index -= 1) {
      const spark = this.sparks[index];
      spark.age += deltaSeconds;
      if (spark.age < 0) continue;
      if (spark.age >= spark.life) {
        this.sparks.splice(index, 1);
        continue;
      }
      spark.previousX = spark.x;
      spark.previousY = spark.y;
      spark.collisionCooldown = Math.max(0, spark.collisionCooldown - deltaSeconds);
      const dragFactor = Math.exp(-spark.drag * deltaSeconds);
      spark.vx *= dragFactor;
      spark.vy = spark.vy * dragFactor + spark.gravity * deltaSeconds;
      spark.x += spark.vx * deltaSeconds;
      spark.y += spark.vy * deltaSeconds;

      if (!spark.interactive || collisionBounds.length === 0) {
        spark.touchingHead = false;
        continue;
      }
      let collision: Collision | null = null;
      for (const collisionBoundsSample of collisionBounds) {
        collision = findEllipseCollision(spark, collisionBoundsSample);
        if (collision) break;
      }
      if (!collision) {
        spark.touchingHead = false;
        continue;
      }
      // One continuous overlap is one impact. Once the full visible streak
      // has left the head, it becomes hittable again instead of being locked
      // forever after an automatic collision at burst time.
      if (spark.touchingHead || spark.collisionCooldown > 0 || spark.collisionCount >= 3) continue;
      spark.touchingHead = true;
      const normalVelocity = spark.vx * collision.nx + spark.vy * collision.ny;
      const tangentX = spark.vx - normalVelocity * collision.nx;
      const tangentY = spark.vy - normalVelocity * collision.ny;
      const headNormalVelocity = this.headVelocityX * collision.nx
        + this.headVelocityY * collision.ny;
      const tangentDirectionX = -collision.ny;
      const tangentDirectionY = collision.nx;
      const headTangentVelocity = this.headVelocityX * tangentDirectionX
        + this.headVelocityY * tangentDirectionY;
      const incomingSpeed = Math.max(0, -normalVelocity);
      const outwardSpeed = Math.max(0, normalVelocity);
      const headPush = Math.max(0, headNormalVelocity);
      const headSpeed = Math.hypot(this.headVelocityX, this.headVelocityY);
      // A spark entering a resting head is allowed to finish its visual path.
      // Reflection only happens when the viewer deliberately moves the head
      // toward the contact normal, so collisions read as user agency rather
      // than two automatic explosions bouncing off both sides of the face.
      const activeHeadImpact = headPush >= 44 && headSpeed >= 70;
      if (!activeHeadImpact) {
        spark.touchingHead = false;
        continue;
      }
      spark.touchingHead = true;
      const rebound = clamp(
        Math.max(
          225,
          outwardSpeed * 1.14 + headPush * 0.78,
          incomingSpeed * 0.88 + headPush * 0.92,
        ),
        225,
        520,
      );
      const sideImpulse = headTangentVelocity * 0.42 + (Math.random() - 0.5) * 38;
      // A leading-point hit is placed back on the boundary. For a visible
      // tail hit, keep the bright head where it is and only redirect velocity;
      // teleporting the head to the contact point would make the streak snap.
      if (!collision.tailOnly) {
        spark.x = collision.x + collision.nx * (spark.radius + 1.5);
        spark.y = collision.y + collision.ny * (spark.radius + 1.5);
      }
      spark.vx = tangentX * 0.62
        + collision.nx * rebound
        + tangentDirectionX * sideImpulse;
      spark.vy = tangentY * 0.62
        + collision.ny * rebound
        + tangentDirectionY * sideImpulse;
      spark.curveBias = clamp(
        spark.curveBias + (Math.random() - 0.5) * 0.12,
        -0.13,
        0.13,
      );
      spark.life = Math.max(spark.life, spark.age + 0.54);
      spark.collisionCooldown = 0.14;
      spark.collisionCount += 1;
      this.collisionCount += 1;
      this.activeCollisionCount += 1;
      this.flashes.push({
        x: collision.x,
        y: collision.y,
        age: 0,
        life: 0.19,
        radius: 11 + spark.radius * 1.7,
        color: spark.color,
        kind: 'impact',
        nx: collision.nx,
        ny: collision.ny,
        strength: 1,
      });
      while (this.flashes.length > MAX_FLASHES) this.flashes.shift();
    }
  }

  draw(target: CanvasRenderingContext2D): void {
    target.save();
    target.globalCompositeOperation = 'screen';
    for (const flash of this.flashes) {
      const progress = clamp(flash.age / flash.life, 0, 1);
      const radius = flash.radius * (0.40 + progress * 0.90);
      const gradient = target.createRadialGradient(flash.x, flash.y, 0, flash.x, flash.y, radius);
      const isBurst = flash.kind === 'burst';
      // A brighter local ignition, using the same gradient and footprint.
      // Collision flashes, blur passes, lifetime and camera remain unchanged.
      gradient.addColorStop(0, withAlpha(flash.color, 1));
      gradient.addColorStop(isBurst ? 0.20 : 0.16, withAlpha(flash.color, isBurst ? 0.76 : 0.62));
      gradient.addColorStop(isBurst ? 0.44 : 0.40, withAlpha(flash.color, isBurst ? 0.24 : 0.18));
      gradient.addColorStop(1, 'rgba(255,255,255,0)');
      const flashFade = flash.kind === 'burst'
        ? Math.pow(1 - progress, 0.68)
        : 1 - progress;
      target.globalAlpha = flashFade * 0.94 * flash.strength;
      target.fillStyle = gradient;
      if (flash.kind === 'impact') {
        const angle = Math.atan2(flash.ny, flash.nx);
        target.save();
        target.translate(flash.x, flash.y);
        target.rotate(angle);
        target.scale(1.8, 0.68);
        target.translate(-flash.x, -flash.y);
      }
      target.beginPath();
      target.arc(flash.x, flash.y, radius, 0, Math.PI * 2);
      target.fill();
      if (flash.kind === 'impact') target.restore();
    }

    for (const spark of this.sparks) {
      if (spark.age < 0) continue;
      const progress = clamp(spark.age / spark.life, 0, 1);
      const fadeIn = Math.min(1, spark.age / 0.055);
      // Hold the useful visible phase longer so a 100-particle burst actually
      // reads as dense; fading half the particles early made it look like 40.
      // Once the crown is open, opacity falls continuously while drag keeps
      // its radius nearly stable. The firework now disappears by fading as a
      // whole instead of flying apart into a few isolated sparks.
      const fadeOut = progress < 0.62
        ? 1
        : 1 - (progress - 0.62) / 0.38;
      const alpha = fadeIn * fadeOut;
      if (alpha <= 0.015) continue;
      const geometry = sparkTrailGeometry(spark);
      // Size-over-life gives the burst a compact ignition, a readable full
      // body, and a restrained exit without changing the collision radius.
      const bodyScale = 0.82
        + Math.sin(Math.min(1, progress / 0.78) * Math.PI) * 0.20;
      const drawRadius = spark.radius * bodyScale;

      // Tiny detached embers add the platform-like granular finish, but they
      // are computed from the current curve and never become physics objects.
      target.globalAlpha = alpha;
      target.fillStyle = spark.color;
      if (spark.age > 0.09 && progress < 0.84) {
        drawDerivedAfterglow(
          target,
          geometry,
          drawRadius,
          spark.detailPhase,
          alpha,
        );
      }

      // A single continuous tapered light trail: narrow at the rear and
      // gently wider at the rounded leading end. Do not stack a second body
      // around it; the only additional mark is a small white-hot leading core.
      target.globalAlpha = alpha;
      if (spark.glow > 0) {
        target.shadowColor = spark.color;
        target.shadowBlur = spark.glow;
      }
      traceTaperedSpark(target, geometry, drawRadius);
      target.fill();
      if (spark.glow > 0) target.shadowBlur = 0;

      if (spark.glow === 0) {
        // Lift the existing leading highlight, not the full ribbon or blur.
        // This adds no draw calls, particles, or changes to the fade timing.
        target.globalAlpha = alpha;
        target.fillStyle = '#fffdf4';
        target.beginPath();
        target.arc(
          geometry.headX,
          geometry.headY,
          Math.max(0.80, drawRadius * 0.48),
          0,
          Math.PI * 2,
        );
        target.fill();
      }
    }
    target.restore();
  }
}
