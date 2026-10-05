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
  collisionCooldown: number;
  collisionCount: number;
  touchingHead: boolean;
};

type BurstFlash = {
  x: number;
  y: number;
  age: number;
  life: number;
  radius: number;
  color: string;
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

const PEARL_PALETTE = ['#fffaff', '#ded2ff', '#ff9ee2', '#c6b8ff'];
const GOLD_PALETTE = ['#fff2bd', '#ffd86b'];
const MAX_SPARKS = 220;
const MAX_FLASHES = 4;
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

function chooseColor(index: number, goldShare: number): string {
  if ((index * 17) % 100 < goldShare * 100) return GOLD_PALETTE[index % GOLD_PALETTE.length];
  return PEARL_PALETTE[index % PEARL_PALETTE.length];
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

export class ProceduralFireworkSystem {
  private sparks: Spark[] = [];
  private flashes: BurstFlash[] = [];
  private sequence = 0;
  private previousHeadCenter: { x: number; y: number } | null = null;
  private previousCollisionBounds: FireworkHeadBounds | null = null;
  private headSampleAge = 0;
  private headVelocityX = 0;
  private headVelocityY = 0;
  collisionCount = 0;

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
    this.collisionCount = 0;
  }

  spawn(bounds: FireworkHeadBounds | null, viewport: FireworkViewport, mode: 'entry' | 'sustain'): void {
    const side = this.sequence % 2 === 0 ? -1 : 1;
    const isEntry = mode === 'entry';
    this.sequence += 1;
    const mainX = bounds
      ? clamp(bounds.cx + side * bounds.rx * 0.92, viewport.width * 0.12, viewport.width * 0.88)
      : viewport.width * (side < 0 ? 0.38 : 0.62);
    const mainY = bounds
      ? clamp(bounds.cy - bounds.ry * 1.08, viewport.height * 0.10, viewport.height * 0.34)
      : viewport.height * 0.24;

    const emitBurst = (
      originX: number,
      originY: number,
      count: number,
      scale: number,
      goldShare: number,
      colorOffset: number,
    ): void => {
      const rotationOffset = Math.random() * Math.PI * 2;
      for (let index = 0; index < count; index += 1) {
        const progress = index / count;
        // Stratified angles retain a legible radial explosion, while broader
        // jitter and continuous speeds remove the synthetic dashed-ring look.
        const angle = rotationOffset + progress * Math.PI * 2 + (Math.random() - 0.5) * 0.34;
        const majorSpark = Math.random() > 0.32;
        // Keep expansion deliberate: the larger spatial scale comes from a
        // slightly longer flight, not a faster blast that disappears sooner.
        const speed = (majorSpark
          ? 188 + Math.pow(Math.random(), 0.62) * 128
          : 112 + Math.random() * 98) * scale * 0.91;
        // This radius is the actual bright particle body. It is intentionally
        // larger instead of faking weight with a thick translucent outline.
        const radius = (majorSpark ? 1.95 + Math.random() * 1.05 : 1.15 + Math.random() * 0.66) * Math.sqrt(scale);
        const color = chooseColor(index + this.sequence * 7 + colorOffset, goldShare);
        this.sparks.push({
          x: originX,
          y: originY,
          previousX: originX,
          previousY: originY,
          vx: Math.cos(angle) * speed,
          vy: Math.sin(angle) * speed - (majorSpark ? 16 : 4) * scale,
          radius,
          trail: (majorSpark ? 28 + Math.random() * 18 : 16 + Math.random() * 12) * Math.sqrt(scale),
          age: -Math.random() * 0.035,
          life: majorSpark ? 0.96 + Math.random() * 0.30 : 0.68 + Math.random() * 0.28,
          gravity: (majorSpark ? 88 + Math.random() * 28 : 98) * scale,
          drag: majorSpark ? 1.85 : 2.15,
          curveBias: (Math.random() - 0.5) * 0.09,
          color,
          collisionCooldown: 0,
          collisionCount: 0,
          touchingHead: false,
        });
      }
      this.flashes.push({
        x: originX,
        y: originY,
        age: 0,
        life: 0.30,
        radius: (isEntry ? 36 : 28) * scale,
        color: goldShare > 0.5
          ? GOLD_PALETTE[this.sequence % GOLD_PALETTE.length]
          : PEARL_PALETTE[this.sequence % PEARL_PALETTE.length],
      });
    };

    emitBurst(mainX, mainY, isEntry ? 124 : 88, 1.42, isEntry ? 0.13 : 0.24, 0);
    if (isEntry) {
      const accentX = bounds
        ? clamp(bounds.cx - side * bounds.rx * 1.08, viewport.width * 0.10, viewport.width * 0.90)
        : viewport.width * (side < 0 ? 0.64 : 0.36);
      const accentY = bounds
        ? clamp(bounds.cy - bounds.ry * 0.28, viewport.height * 0.18, viewport.height * 0.58)
        : viewport.height * 0.42;
      emitBurst(accentX, accentY, 52, 0.78, 0.72, 41);
    }
    if (this.sparks.length > MAX_SPARKS) this.sparks.splice(0, this.sparks.length - MAX_SPARKS);
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

  update(deltaSeconds: number, bounds: FireworkHeadBounds | null): void {
    const collisionBounds = this.collisionBoundsForFrame(bounds);
    this.updateHeadVelocity(deltaSeconds, bounds);
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

      if (collisionBounds.length === 0) {
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
      const rebound = clamp(
        Math.max(
          170,
          outwardSpeed * 1.14 + headPush * 0.78,
          incomingSpeed * 0.88 + headPush * 0.92,
        ),
        170,
        480,
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
      spark.life = Math.max(spark.life, spark.age + 0.42);
      spark.collisionCooldown = 0.14;
      spark.collisionCount += 1;
      this.collisionCount += 1;
      this.flashes.push({
        x: collision.x,
        y: collision.y,
        age: 0,
        life: 0.16,
        radius: 8 + spark.radius * 1.8,
        color: spark.color,
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
      gradient.addColorStop(0, withAlpha(flash.color, 0.92));
      gradient.addColorStop(0.22, withAlpha(flash.color, 0.34));
      gradient.addColorStop(1, 'rgba(255,255,255,0)');
      target.globalAlpha = (1 - progress) * 0.82;
      target.fillStyle = gradient;
      target.beginPath();
      target.arc(flash.x, flash.y, radius, 0, Math.PI * 2);
      target.fill();
    }

    for (const spark of this.sparks) {
      if (spark.age < 0) continue;
      const progress = clamp(spark.age / spark.life, 0, 1);
      const alpha = Math.min(1, spark.age / 0.050) * Math.pow(1 - progress, 0.82);
      if (alpha <= 0.015) continue;
      const geometry = sparkTrailGeometry(spark);

      // A single continuous tapered light trail: narrow at the rear and
      // gently wider at the rounded leading end. Do not stack a second body
      // inside it; that made each spark read as two fish scales instead of one
      // clean firework particle.
      target.globalAlpha = alpha;
      target.fillStyle = spark.color;
      traceTaperedSpark(target, geometry, spark.radius);
      target.fill();
    }
    target.restore();
  }
}
