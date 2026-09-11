import { FilesetResolver, HandLandmarker }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

// Draw uses hysteresis: tighter to start, looser to stop — prevents flickering.
// Measured as a fraction of palm length, in 3D, so it doesn't change when the hand
// moves nearer/further or tilts (fixed pixels broke lines drawn across the face/body).
const PINCH_START_RATIO = 0.25;
const PINCH_KEEP_RATIO  = 0.45;
const COLOR_THRESHOLD      = 60;
const UNDO_THRESHOLD       = 60;

// Minimum pixel distance between recorded points — avoids Catmull-Rom overshoot
const MIN_POINT_DIST = 4;

// Tracking drops out for a frame or two whenever the lighting shifts. Don't end a
// stroke on the first bad frame: hold it open this long, and if the pinch comes
// back in time, keep drawing into the same line.
const STROKE_GRACE_MS = 300;
// ...unless the hand reappears this far from where the line stopped — then it's a new line.
const MAX_BRIDGE_DIST = 150;
// A fist has to hold this many frames before it counts, so one misread frame
// can't cut a line off.
const FIST_CONFIRM_FRAMES = 3;
// Exponential smoothing on the fingertip position and pinch distance (0–1, lower = smoother)
const SMOOTH_POS   = 0.5;
const SMOOTH_PINCH = 0.4;

// ── Lighting normalisation ──
// The model sees a contrast-stretched copy of each frame, so a lamp switching on
// or a cloud passing doesn't change what the hand looks like to it. The display
// still shows the raw camera. Add ?norm=0 to the URL to turn it off and compare.
const NORMALISE      = new URLSearchParams(location.search).get('norm') !== '0';
const NORM_WIDTH     = 320;   // the model downsamples anyway; this keeps pixel work cheap
const NORM_LOW_PCT   = 0.02;  // black point: darkest 2% of pixels
const NORM_HIGH_PCT  = 0.98;  // white point: brightest 2%
const NORM_ADAPT     = 0.15;  // how fast the levels follow the light (0–1), slow enough not to pump
const NORM_MIN_RANGE = 64;    // caps the gain at 4x so a dark room turns grainy, not noise
let normCanvas = null, normCtx = null;
let normLo = 0, normHi = 255;

function normalisedFrame(videoEl) {
  const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
  if (!vw || !vh) return videoEl;
  const w = NORM_WIDTH, h = Math.round(NORM_WIDTH * vh / vw);
  if (!normCanvas || normCanvas.height !== h) {
    normCanvas = document.createElement('canvas');
    normCanvas.width = w; normCanvas.height = h;
    normCtx = normCanvas.getContext('2d', { willReadFrequently: true });
  }
  normCtx.drawImage(videoEl, 0, 0, w, h);
  const img = normCtx.getImageData(0, 0, w, h);
  const px = img.data, n = w * h;

  // Luminance histogram → this frame's black and white points
  const hist = new Uint32Array(256);
  for (let i = 0; i < px.length; i += 4) {
    hist[(px[i] * 77 + px[i + 1] * 150 + px[i + 2] * 29) >> 8]++;
  }
  let lo = 0, hi = 255, acc = 0;
  for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc >= n * NORM_LOW_PCT) { lo = v; break; } }
  acc = 0;
  for (let v = 255; v >= 0; v--) { acc += hist[v]; if (acc >= n * (1 - NORM_HIGH_PCT)) { hi = v; break; } }

  // Ease towards them rather than jumping, so the image doesn't flicker frame to frame
  normLo += (lo - normLo) * NORM_ADAPT;
  normHi += (hi - normHi) * NORM_ADAPT;
  let range = normHi - normLo;
  let base = normLo;
  if (range < NORM_MIN_RANGE) { base -= (NORM_MIN_RANGE - range) / 2; range = NORM_MIN_RANGE; }

  // Stretch every channel by the same amount so skin tone keeps its colour
  const lut = new Uint8ClampedArray(256);
  for (let v = 0; v < 256; v++) lut[v] = ((v - base) * 255) / range;
  for (let i = 0; i < px.length; i += 4) {
    px[i] = lut[px[i]]; px[i + 1] = lut[px[i + 1]]; px[i + 2] = lut[px[i + 2]];
  }
  normCtx.putImageData(img, 0, 0);
  return normCanvas;
}

// Flower particle system
let flowers = [];

// Warm palette: pink, yellow, lavender, coral, mint
const FLOWER_COLORS = [
  [255, 160, 200],
  [255, 215,  90],
  [190, 155, 245],
  [255, 130, 110],
  [130, 215, 175],
];

function makeFlower(x, y, vx, vy, gravity, decay) {
  const c = FLOWER_COLORS[floor(random(FLOWER_COLORS.length))];
  return { x, y, vx, vy, gravity, rot: random(TWO_PI), rotV: random(-0.1, 0.1),
           size: random(14, 26), life: 1.0, decay, r: c[0], g: c[1], b: c[2] };
}

function spawnFlowerBurst(cx, cy) {
  for (let i = 0; i < 22; i++) {
    const angle = random(-PI, 0);
    const speed = random(4, 11);
    flowers.push(makeFlower(
      cx + random(-40, 40), cy,
      cos(angle) * speed, sin(angle) * speed,
      0.28, random(0.01, 0.016)
    ));
  }
}

function spawnFlowerRainDrop() {
  flowers.push(makeFlower(
    random(width), -30,
    random(-1.2, 1.2), random(2.5, 5),
    0.04, 0.005
  ));
}

// Wave tracking — detect direction reversals in wrist x over a short window
const wristXHistory = [];
const WAVE_HISTORY  = 10;
const WAVE_MOVE_MIN = 0.015; // normalized units — ~1.5% of frame width per frame

function recordWrist(h) {
  wristXHistory.push(h[0].x);
  if (wristXHistory.length > WAVE_HISTORY) wristXHistory.shift();
}

function isWaving() {
  if (wristXHistory.length < WAVE_HISTORY) return false;
  let reversals = 0, prevDir = null;
  for (let i = 1; i < wristXHistory.length; i++) {
    const v = wristXHistory[i] - wristXHistory[i - 1];
    if (Math.abs(v) < WAVE_MOVE_MIN) continue;
    const dir = v > 0 ? 1 : -1;
    if (prevDir !== null && dir !== prevDir) reversals++;
    prevDir = dir;
  }
  return reversals >= 2;
}

// Pinch as a fraction of palm length, in 3D. Landmarks are normalised to the
// frame (x to width, y to height, z roughly on x's scale), so scale back to the
// video's aspect before measuring or distances stretch with the frame shape.
function pinchRatio(h) {
  const vw = video?.elt?.videoWidth || width, vh = video?.elt?.videoHeight || height;
  const d3 = (a, b) => Math.hypot((a.x - b.x) * vw, (a.y - b.y) * vh, ((a.z ?? 0) - (b.z ?? 0)) * vw);
  // Palm length from the wrist to the index and pinky knuckles; whichever is
  // longer survives the hand turning side-on
  const palm = Math.max(d3(h[0], h[5]), d3(h[0], h[17]));
  if (palm < 1) return Infinity;
  // Closest thumb-to-index contact of three pairs, so a single misplaced
  // fingertip doesn't read as the pinch opening
  const gap = Math.min(d3(h[4], h[8]), d3(h[4], h[7]), d3(h[3], h[8]));
  return gap / palm;
}

// Fist: all four fingertips below their PIP joints (y increases downward)
const FIST_MARGIN = 0.02;
function isFist(h) {
  return [[8,6],[12,10],[16,14],[20,18]]
    .every(([tip, pip]) => h[tip].y > h[pip].y + FIST_MARGIN);
}

function drawFlower(f) {
  const a   = max(0, f.life) * 255;
  const r   = f.size * 0.5;
  push();
  translate(f.x, f.y);
  rotate(f.rot);
  noStroke();
  // 5 petals
  fill(f.r, f.g, f.b, a);
  for (let i = 0; i < 5; i++) {
    push();
    rotate((i / 5) * TWO_PI);
    ellipse(0, r * 0.55, r * 0.55, r);
    pop();
  }
  // centre
  fill(255, 230, 100, a);
  circle(0, 0, r * 0.55);
  pop();
}

let video, handLandmarker;
let ready = false, detecting = false;
let paths = [], currentPath = [];
let isDrawing = false;

let currentColor = '#0066ff';
const colorPalette = ['#0066ff','#ff3366','#ffaa00','#33cc33','#8000ff'];
let currentColorIndex = 0;

let currentBrushSize = 4;
let cameraEnabled = true;
let lastPos = null;
let handVisible = false;

// stroke-smoothing state
let releaseAt   = null;  // when the pinch/hand was first lost mid-stroke
let smoothPos   = null;
let smoothPinch = null;
let fistFrames  = 0;

function endStroke(badge, reason) {
  if (DEBUG && reason && currentPath.length > 1) {
    dbg.breaks[reason] = (dbg.breaks[reason] || 0) + 1;
    dbg.lastBreak = { reason, t: performance.now() };
  }
  if (currentPath.length > 1) paths.push(currentPath.slice());
  currentPath = [];
  isDrawing = false;
  releaseAt = null;
  if (badge && window.setGestureBadge) window.setGestureBadge(badge);
}

// Called on any frame where the stroke would have ended; only ends it once the
// loss has lasted longer than the grace period.
function maybeRelease(now, badge, reason) {
  if (!isDrawing) return;
  if (releaseAt === null) releaseAt = now;
  else if (now - releaseAt > STROKE_GRACE_MS) endStroke(badge, reason);
}

// ── Debug overlay (?debug=1) ──
// Shows what the tracker sees and why each line ended, so breaks can be traced
// to a cause instead of guessed at.
const DEBUG = new URLSearchParams(location.search).get('debug') === '1';
const dbg = { hand: false, score: null, landmarks: null, rawPinch: null, pinch: null, peakPinch: null,
              seen: [], breaks: {}, lastBreak: null, el: null };

function recordDebug(res) {
  dbg.hand = !!res.landmarks?.length;
  dbg.landmarks = dbg.hand ? res.landmarks[0] : null;
  const hd = res.handednesses || res.handedness;
  dbg.score = hd?.[0]?.[0]?.score ?? null;
  if (!dbg.hand) { dbg.rawPinch = null; dbg.pinch = null; }
  dbg.seen.push(dbg.hand);
  if (dbg.seen.length > 60) dbg.seen.shift();  // ~2s at 30fps
}

function drawDebug() {
  const h = dbg.landmarks;
  if (h) {
    const P = (l) => [width - l.x * width, l.y * height];
    noStroke(); fill(255, 255, 255, 200);
    for (const l of h) { const [x, y] = P(l); circle(x, y, 7); }
    const [ix, iy] = P(h[8]), [tx, ty] = P(h[4]);
    const threshold = isDrawing ? PINCH_KEEP_RATIO : PINCH_START_RATIO;
    stroke(dbg.pinch !== null && dbg.pinch < threshold ? color(40, 200, 120) : color(230, 60, 60));
    strokeWeight(3); line(ix, iy, tx, ty);
    noStroke(); fill(255, 220, 0); circle(ix, iy, 12); circle(tx, ty, 12);
  }
  // What the model is actually given (after normalisation), mirrored to match the view
  if (NORMALISE && normCanvas) {
    const tw = 180, th = tw * normCanvas.height / normCanvas.width;
    const ctx = drawingContext;
    ctx.save(); ctx.translate(12 + tw, height - th - 12); ctx.scale(-1, 1);
    ctx.drawImage(normCanvas, 0, 0, tw, th); ctx.restore();
    noFill(); stroke(255); strokeWeight(1); rect(12, height - th - 12, tw, th);
  }

  if (!dbg.el) {
    dbg.el = document.createElement('pre');
    dbg.el.style.cssText = 'position:fixed;top:64px;right:16px;z-index:200;margin:0;padding:10px 12px;' +
      'background:rgba(20,20,20,.82);color:#fff;font:12px/1.5 ui-monospace,Menlo,monospace;' +
      'border-radius:8px;pointer-events:none;white-space:pre';
    document.body.appendChild(dbg.el);
  }
  const seenPct = dbg.seen.length ? Math.round(100 * dbg.seen.filter(Boolean).length / dbg.seen.length) : 0;
  const state = !isDrawing ? 'not drawing'
    : releaseAt !== null ? `holding line (${Math.round(performance.now() - releaseAt)}ms of ${STROKE_GRACE_MS})`
    : 'drawing';
  const breaks = Object.entries(dbg.breaks).map(([r, n]) => `  ${r}: ${n}`).join('\n') || '  none yet';
  const last = dbg.lastBreak
    ? `${dbg.lastBreak.reason}, ${((performance.now() - dbg.lastBreak.t) / 1000).toFixed(1)}s ago` : '-';
  dbg.el.textContent =
    `hand found:   ${dbg.hand ? 'yes' : 'NO'}${dbg.score !== null ? ` (confidence ${dbg.score.toFixed(2)})` : ''}\n` +
    `tracked:      ${seenPct}% of last 2s\n` +
    `pinch:        ${dbg.pinch !== null ? dbg.pinch.toFixed(2) : '-'}` +
    `${dbg.rawPinch !== null ? ` (raw ${dbg.rawPinch.toFixed(2)})` : ''}  start <${PINCH_START_RATIO}, keep <${PINCH_KEEP_RATIO}` +
    `  [x palm length]\n` +
    `peak pinch while drawing: ${dbg.peakPinch !== null ? dbg.peakPinch.toFixed(2) : '-'}\n` +
    `fist frames:  ${fistFrames} (counts at ${FIST_CONFIRM_FRAMES})\n` +
    `state:        ${state}\n` +
    `normalise:    ${NORMALISE ? 'on' : 'off'}\n` +
    `lines ended by:\n${breaks}\n` +
    `last break:   ${last}`;
}

// rising-edge flags
let colorGestureActive = false;
let undoGestureActive  = false;
let fistGestureActive  = false;
let waveGestureActive  = false;

// Update color swatches in the toolbar
function updateSwatches() {
  document.querySelectorAll('#palette .swatch').forEach((el, i) => {
    el.classList.toggle('active', i === currentColorIndex);
  });
}

async function setup() {
  const wrapper = document.getElementById('canvas-wrapper');
  const cnv = createCanvas(wrapper.offsetWidth, wrapper.offsetHeight);
  cnv.parent(wrapper);
  frameRate(30);

  const size = min(windowWidth, windowHeight) * 0.9;
  video = createCapture(VIDEO, () => console.log('Camera started'));
  video.size(size, size);
  video.hide();

  // Brush size slider — keep internal state in sync
  const brushSlider = document.getElementById('brushSize');
  brushSlider.addEventListener('input', () => {
    currentBrushSize = parseInt(brushSlider.value, 10);
  });

  select('#toggleCam').mousePressed(toggleCamera);
  select('#undoBtn').mousePressed(() => { if (paths.length) paths.pop(); });
  select('#clearBtn').mousePressed(() => {
    paths = []; currentPath = []; flowers = []; isDrawing = false; releaseAt = null;
  });
  select('#saveBtn').mousePressed(saveAsSVG);

  await loadModel();
}

async function loadModel() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: './hand_landmarker.task' },
    runningMode: 'VIDEO', numHands: 1,
    // Defaults are 0.5; lower keeps hold of the hand through dim or uneven light
    minHandDetectionConfidence: 0.3,
    minHandPresenceConfidence:  0.3,
    minTrackingConfidence:      0.3
  });
  ready = true;

  // Remove loader once model is ready
  const loader = document.getElementById('loader');
  if (loader) loader.remove();
}

function draw() {
  clear();
  if (cameraEnabled) {
    push(); translate(width, 0); scale(-1, 1); image(video, 0, 0, width, height); pop();
  } else if (lastPos) {
    noStroke(); fill(currentColor);
    circle(lastPos.x, lastPos.y, currentBrushSize * 2);
  }

  if (ready && !detecting) trackHand();

  noFill();
  const allPaths = currentPath.length > 1 ? [...paths, currentPath] : paths;
  for (const path of allPaths) {
    if (path.length < 2) continue;
    stroke(path[0].color);
    strokeWeight(path[0].size);
    const first = path[0].point, last = path[path.length - 1].point;
    beginShape();
    curveVertex(first.x, first.y);
    for (const pt of path) curveVertex(pt.point.x, pt.point.y);
    curveVertex(last.x, last.y);
    endShape();
  }

  // Flower particles
  flowers = flowers.filter(f => f.life > 0 && f.y < height + 60);
  for (const f of flowers) {
    f.vy += f.gravity;
    f.x  += f.vx;
    f.y  += f.vy;
    f.rot += f.rotV;
    f.life -= f.decay;
    drawFlower(f);
  }

  if (DEBUG) drawDebug();
}

async function trackHand() {
  detecting = true;
  try {
    const now = performance.now();
    const frame = NORMALISE ? normalisedFrame(video.elt) : video.elt;
    const res = await handLandmarker.detectForVideo(frame, now);
    if (DEBUG) recordDebug(res);

    if (res.landmarks?.length) {
      if (!handVisible) {
        handVisible = true;
        if (window.setGestureBadge) window.setGestureBadge('hand');
      }

      const h    = res.landmarks[0];
      const idx  = h[8], thumb = h[4], ring = h[16], pinky = h[20];
      const toPx = (a, b) => dist((a.x - b.x) * width, (a.y - b.y) * height, 0, 0);
      const rawPinch = pinchRatio(h);
      const dColor = toPx(ring, thumb);
      const dUndo  = toPx(pinky, thumb);

      // mirror coords, then smooth so one jittery frame doesn't jolt the line
      const rawX = width - idx.x * width;
      const rawY = idx.y * height;
      if (smoothPos) {
        smoothPos.x = lerp(smoothPos.x, rawX, SMOOTH_POS);
        smoothPos.y = lerp(smoothPos.y, rawY, SMOOTH_POS);
      } else {
        smoothPos = createVector(rawX, rawY);
      }
      smoothPinch = smoothPinch === null ? rawPinch : lerp(smoothPinch, rawPinch, SMOOTH_PINCH);
      const x = smoothPos.x, y = smoothPos.y;
      const dDraw = smoothPinch;
      lastPos = createVector(x, y);

      const fist = isFist(h);
      fistFrames = fist ? fistFrames + 1 : 0;
      if (DEBUG) {
        dbg.rawPinch = rawPinch; dbg.pinch = dDraw;
        // Highest pinch reading during the current line — shows how close a break came
        if (isDrawing) dbg.peakPinch = Math.max(dbg.peakPinch ?? 0, dDraw);
        else dbg.peakPinch = null;
      }

      // FIST takes priority — checked first so it isn't blocked by draw detection
      // (making a fist also brings thumb+index together, which would trigger draw)
      if (fistFrames >= FIST_CONFIRM_FRAMES) {
        // cancel any in-progress stroke
        if (isDrawing) endStroke(null, 'fist detected');
        if (!fistGestureActive) {
          fistGestureActive = true;
          const hx = width - h[9].x * width;
          const hy = h[9].y * height;
          spawnFlowerBurst(hx, hy);
          if (window.setGestureBadge) window.setGestureBadge('fist');
        }
      } else if (fist) {
        // Looks like a fist but not confirmed yet — hold the stroke as it is
        // rather than drawing or ending it on a possibly misread frame.
      } else {
        fistGestureActive = false;

        // DRAW: hysteresis — tight threshold to start, relaxed to stop
        const stopThreshold = isDrawing ? PINCH_KEEP_RATIO : PINCH_START_RATIO;
        if (dDraw < stopThreshold) {
          const last = currentPath[currentPath.length - 1];
          // Coming back from a dropout far from where the line stopped: that's a new line
          if (isDrawing && releaseAt !== null && last &&
              dist(x, y, last.point.x, last.point.y) > MAX_BRIDGE_DIST) {
            endStroke(null, 'hand came back far away');
          }
          releaseAt = null;
          if (!isDrawing) {
            isDrawing = true;
            currentPath = [];
            if (window.setGestureBadge) window.setGestureBadge('drawing');
          }
          const prev = currentPath[currentPath.length - 1];
          if (!prev || dist(x, y, prev.point.x, prev.point.y) >= MIN_POINT_DIST) {
            currentPath.push({ point: createVector(x, y), color: currentColor, size: currentBrushSize });
          }
        } else {
          maybeRelease(now, 'hand', 'pinch read as open');
        }

        // COLOR: thumb+ring rising-edge, not while drawing
        if (!isDrawing && dColor < COLOR_THRESHOLD) {
          if (!colorGestureActive) {
            colorGestureActive = true;
            currentColorIndex = (currentColorIndex + 1) % colorPalette.length;
            currentColor = colorPalette[currentColorIndex];
            updateSwatches();
            if (window.setGestureBadge) window.setGestureBadge('color');
          }
        } else {
          colorGestureActive = false;
        }

        // UNDO: thumb+pinky rising-edge, not while drawing
        if (!isDrawing && dUndo < UNDO_THRESHOLD) {
          if (!undoGestureActive && paths.length) {
            undoGestureActive = true;
            paths.pop();
            if (window.setGestureBadge) window.setGestureBadge('undo');
          }
        } else {
          undoGestureActive = false;
        }
      }

      // WAVE: runs independently of fist/draw — tracks wrist movement history
      recordWrist(h);
      if (isWaving()) {
        if (!waveGestureActive) {
          waveGestureActive = true;
          if (window.setGestureBadge) window.setGestureBadge('wave');
        }
        for (let i = 0; i < 3; i++) spawnFlowerRainDrop();
      } else {
        waveGestureActive = false;
      }

    } else {
      if (handVisible) {
        handVisible = false;
        if (window.setGestureBadge) window.setGestureBadge('idle');
      }
      fistFrames = 0;
      maybeRelease(now, null, 'hand lost by tracker');
      // Once the stroke is really over, drop the smoothing history so the next
      // line doesn't glide in from where the last one ended.
      if (!isDrawing) { smoothPos = null; smoothPinch = null; }
    }
  } catch (e) {
    console.error(e);
  }
  detecting = false;
}

function toggleCamera() {
  cameraEnabled = !cameraEnabled;
  select('#toggleCam').addClass(cameraEnabled ? '' : 'cam-off');
  select('#toggleCam').removeClass(cameraEnabled ? 'cam-off' : '');
}

function saveAsSVG() {
  let svg = `<svg xmlns='http://www.w3.org/2000/svg' width='${width}' height='${height}'>`;
  for (const path of paths) {
    svg += `<polyline fill='none' stroke='${path[0].color}' stroke-width='${path[0].size}' points='`;
    svg += path.map(p => `${p.point.x},${p.point.y}`).join(' ');
    svg += `'/>\n`;
  }
  svg += `</svg>`;
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }));
  link.download = 'gesture_drawing.svg';
  link.click();
}

// expose to p5.js
window.setup = setup;
window.draw  = draw;
