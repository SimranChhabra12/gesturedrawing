import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

let video;
let handLandmarker;
let ready = false;
let paths = [];
let currentPath = [];
let isDrawing = false;
let detecting = false;

let currentColor = '#0066ff';
const colorPalette = ['#0066ff', '#ff3366', '#ffaa00', '#33cc33', '#8000ff'];
let currentColorIndex = 0;

async function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();

  document.getElementById("saveBtn").addEventListener("click", saveAsSVG);
  document.getElementById("colorBtn").addEventListener("click", () => {
    currentColorIndex = (currentColorIndex + 1) % colorPalette.length;
    currentColor = colorPalette[currentColorIndex];
  });

  await loadModel();
  startHandTrackingLoop();
}

async function loadModel() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "./hand_landmarker.task"
    },
    runningMode: "VIDEO",
    numHands: 1
  });

  ready = true;
}

function startHandTrackingLoop() {
  setInterval(trackHand, 150); // Reduce frequency to avoid overload
}

async function trackHand() {
  if (!ready || !video.loadedmetadata || detecting) return;
  detecting = true;

  try {
    const now = performance.now();
    const result = await handLandmarker.detectForVideo(video.elt, now);

    if (result.landmarks?.length > 0) {
      const hand = result.landmarks[0];
      const indexTip = hand[8];
      const thumbTip = hand[4];
      const middleTip = hand[12];

      const drawDist = dist(indexTip.x, indexTip.y, thumbTip.x, thumbTip.y);
      const colorDist = dist(middleTip.x, middleTip.y, thumbTip.x, thumbTip.y);

      const x = width - indexTip.x * width;
      const y = indexTip.y * height;
      const point = createVector(x, y);

      // Gesture to change color (thumb + middle finger)
      if (colorDist < 0.05) {
        currentColorIndex = (currentColorIndex + 1) % colorPalette.length;
        currentColor = colorPalette[currentColorIndex];
      }

      // Gesture to draw (thumb + index finger close)
      if (drawDist < 0.05) {
        isDrawing = true;
        currentPath.push({ point, color: currentColor });
      } else {
        if (isDrawing && currentPath.length > 0) {
          paths.push([...currentPath]);
          currentPath = [];
        }
        isDrawing = false;
      }
    } else {
      if (isDrawing && currentPath.length > 0) {
        paths.push([...currentPath]);
        currentPath = [];
      }
      isDrawing = false;
    }
  } catch (error) {
    console.error("MediaPipe detection failed:", error);
  }

  detecting = false;
}

function draw() {
  background(255);

  if (video && video.loadedmetadata) {
    push();
    translate(width, 0);
    scale(-1, 1);
    image(video, 0, 0, width, height);
    pop();
  }

  noFill();
  strokeWeight(4);

  for (let path of paths) {
    beginShape();
    stroke(path[0]?.color || '#0066ff');
    for (let { point } of path) {
      curveVertex(point.x, point.y);
    }
    endShape();
  }

  if (currentPath.length > 0) {
    beginShape();
    stroke(currentPath[0]?.color || '#0066ff');
    for (let { point } of currentPath) {
      curveVertex(point.x, point.y);
    }
    endShape();
  }
}

function saveAsSVG() {
  let svg = `<svg xmlns='http://www.w3.org/2000/svg' width='640' height='480'>`;
  for (let path of paths) {
    if (path.length > 0) {
      svg += `<polyline fill='none' stroke='${path[0].color}' stroke-width='4' points='`;
      svg += path.map(({ point }) => `${point.x},${point.y}`).join(' ');
      svg += `'/>\n`;
    }
  }
  svg += `</svg>`;

  const blob = new Blob([svg], { type: 'image/svg+xml' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'gesture_drawing.svg';
  link.click();
}

window.setup = setup;
window.draw = draw;
