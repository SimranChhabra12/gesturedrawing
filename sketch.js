import { FilesetResolver, HandLandmarker }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

// Draw uses hysteresis: tighter to start, looser to stop — prevents flickering
const DRAW_START_THRESHOLD = 45;
const DRAW_STOP_THRESHOLD  = 72;
const COLOR_THRESHOLD      = 60;
const UNDO_THRESHOLD       = 60;

// Minimum pixel distance between recorded points — avoids Catmull-Rom overshoot
const MIN_POINT_DIST = 4;

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
  select('#clearBtn').mousePressed(() => { paths = []; currentPath = []; flowers = []; });
  select('#saveBtn').mousePressed(saveAsSVG);

  await loadModel();
}

async function loadModel() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: './hand_landmarker.task' },
    runningMode: 'VIDEO', numHands: 1
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
}

async function trackHand() {
  detecting = true;
  try {
    const now = performance.now();
    const res = await handLandmarker.detectForVideo(video.elt, now);

    if (res.landmarks?.length) {
      if (!handVisible) {
        handVisible = true;
        if (window.setGestureBadge) window.setGestureBadge('hand');
      }

      const h    = res.landmarks[0];
      const idx  = h[8], thumb = h[4], ring = h[16], pinky = h[20];
      const toPx = (a, b) => dist((a.x - b.x) * width, (a.y - b.y) * height, 0, 0);
      const dDraw  = toPx(idx, thumb);
      const dColor = toPx(ring, thumb);
      const dUndo  = toPx(pinky, thumb);

      // mirror coords
      const x = width - idx.x * width;
      const y = idx.y * height;
      lastPos = createVector(x, y);

      // FIST takes priority — checked first so it isn't blocked by draw detection
      // (making a fist also brings thumb+index together, which would trigger draw)
      if (isFist(h)) {
        // cancel any in-progress stroke
        if (isDrawing) {
          if (currentPath.length > 1) paths.push(currentPath.slice());
          currentPath = [];
          isDrawing = false;
        }
        if (!fistGestureActive) {
          fistGestureActive = true;
          const hx = width - h[9].x * width;
          const hy = h[9].y * height;
          spawnFlowerBurst(hx, hy);
          if (window.setGestureBadge) window.setGestureBadge('fist');
        }
      } else {
        fistGestureActive = false;

        // DRAW: hysteresis — tight threshold to start, relaxed to stop
        const stopThreshold = isDrawing ? DRAW_STOP_THRESHOLD : DRAW_START_THRESHOLD;
        if (dDraw < stopThreshold) {
          if (!isDrawing) {
            isDrawing = true;
            currentPath = [];
            if (window.setGestureBadge) window.setGestureBadge('drawing');
          }
          const last = currentPath[currentPath.length - 1];
          if (!last || dist(x, y, last.point.x, last.point.y) >= MIN_POINT_DIST) {
            currentPath.push({ point: createVector(x, y), color: currentColor, size: currentBrushSize });
          }
        } else if (isDrawing) {
          if (currentPath.length > 1) paths.push(currentPath.slice());
          currentPath = [];
          isDrawing = false;
          if (window.setGestureBadge) window.setGestureBadge('hand');
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
      if (isDrawing) {
        if (currentPath.length > 1) paths.push(currentPath.slice());
        currentPath = [];
        isDrawing = false;
      }
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
