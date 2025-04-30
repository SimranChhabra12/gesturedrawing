import { FilesetResolver, HandLandmarker }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

// Gesture thresholds in px
const DRAW_THRESHOLD  = 60;  // thumb+index pinch
const COLOR_THRESHOLD = 60;  // thumb+ring pinch
const UNDO_THRESHOLD  = 60;  // thumb+pinky pinch

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

// rising-edge flags
let colorGestureActive = false;
let undoGestureActive  = false;

async function setup() {
  const size = min(windowWidth, windowHeight) * 0.9;
  createCanvas(min(windowWidth, windowHeight) * 0.99, min(windowWidth, windowHeight) * 0.99);
  frameRate(30);

  video = createCapture(VIDEO, () => console.log('Camera started'));
  video.size(size, size);
  video.hide();

  // brush-size slider
  const brushSlider = document.getElementById('brushSize');
  brushSlider.addEventListener('input', () => {
    currentBrushSize = parseInt(brushSlider.value, 10);
  });

  select('#toggleCam').mousePressed(toggleCamera);
  select('#undoBtn').mousePressed(() => { if (paths.length) paths.pop(); });
  select('#clearBtn').mousePressed(() => { paths = []; currentPath = []; });
  select('#saveBtn').mousePressed(saveAsSVG);
  select('#colorIndicator').style('background', currentColor);

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

  // draw saved paths
  noFill();
  for (const path of paths) {
    stroke(path[0].color);
    strokeWeight(path[0].size);
    beginShape(); path.forEach(pt => curveVertex(pt.point.x, pt.point.y)); endShape();
  }

  // draw current path
  if (currentPath.length) {
    stroke(currentPath[0].color);
    strokeWeight(currentPath[0].size);
    beginShape(); currentPath.forEach(pt => curveVertex(pt.point.x, pt.point.y)); endShape();
  }
}

async function trackHand() {
  detecting = true;
  try {
    const now = performance.now();
    const res = await handLandmarker.detectForVideo(video.elt, now);
    if (res.landmarks?.length) {
      const h    = res.landmarks[0];
      const idx  = h[8], thumb = h[4], ring = h[16], pinky = h[20];
      const toPx = (a,b) => dist((a.x-b.x)*width, (a.y-b.y)*height, 0, 0);
      const dDraw  = toPx(idx, thumb);
      const dColor = toPx(ring, thumb);
      const dUndo  = toPx(pinky, thumb);

      // mirror coords
      const x = width - idx.x * width;
      const y = idx.y * height;
      lastPos = createVector(x, y);

      // DRAW: thumb+index pinch
      if (dDraw < DRAW_THRESHOLD) {
        if (!isDrawing) {
          isDrawing = true;
          currentPath = [];
        }
        currentPath.push({ point: createVector(x, y), color: currentColor, size: currentBrushSize });
      } else if (isDrawing) {
        if (currentPath.length > 1) paths.push(currentPath.slice());
        currentPath = [];
        isDrawing = false;
      }

      // COLOR: thumb+ring rising-edge only when not drawing
      if (!isDrawing && dColor < COLOR_THRESHOLD) {
        if (!colorGestureActive) {
          colorGestureActive = true;
          currentColorIndex = (currentColorIndex + 1) % colorPalette.length;
          currentColor = colorPalette[currentColorIndex];
          select('#colorIndicator').style('background', currentColor);
        }
      } else {
        colorGestureActive = false;
      }

      // UNDO: thumb+pinky rising-edge only when not drawing
      if (!isDrawing && dUndo < UNDO_THRESHOLD) {
        if (!undoGestureActive && paths.length) {
          undoGestureActive = true;
          paths.pop();
        }
      } else {
        undoGestureActive = false;
      }

    } else if (isDrawing) {
      if (currentPath.length > 1) paths.push(currentPath.slice());
      currentPath = [];
      isDrawing = false;
    }
  } catch (e) {
    console.error(e);
  }
  detecting = false;
}

function toggleCamera() {
  cameraEnabled = !cameraEnabled;
  select('#toggleCam').html(cameraEnabled ? 'Disable Camera' : 'Enable Camera');
}

function saveAsSVG() {
  let svg = `<svg xmlns='http://www.w3.org/2000/svg' width='${width}' height='${height}'>`;
  for (const path of paths) {
    svg += `<polyline fill='none' stroke='${path[0].color}' stroke-width='${path[0].size}' points='`;
    svg += path.map(p => `${p.point.x},${p.point.y}`).join(' ');
    svg += `'/>
`;
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