import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

let video;
let handLandmarker;
let drawing = [];
let lastPoint = null;
let ready = false;
let brushSizeSlider;

// Recording variables
let mediaRecorder;
let recordedChunks = [];
let recording = false;

async function setup() {
  createCanvas(640, 480).parent(document.body);
  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();
  
  // Setup UI elements
  document.getElementById('clearBtn').addEventListener('click', () => {
    drawing = [];
  });
  brushSizeSlider = document.getElementById('brushSize');
  
  const recordBtn = document.getElementById('recordBtn');
  const downloadLink = document.getElementById('downloadLink');
  recordBtn.addEventListener('click', () => {
    if (!recording) {
      startRecording();
      recordBtn.textContent = 'Stop Recording';
    } else {
      stopRecording(() => {
        recordBtn.textContent = 'Start Recording';
        downloadLink.style.display = 'inline';
      });
    }
  });
  
  // Load the hand tracking model
  await loadModel();
  // Start hand tracking at regular intervals (every 100ms)
  setInterval(trackHand, 100);
}

async function loadModel() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      // Update this path to point to your locally downloaded file.
      // You can use a relative path if you move it into your project folder, e.g., "./hand_landmarker.task"
      modelAssetPath: "./hand_landmarker.task"
    },
    runningMode: "VIDEO",
    numHands: 1
  });
  ready = true;
}

async function trackHand() {
  if (!ready || !video.loadedmetadata) return;
  
  const now = performance.now();
  const result = await handLandmarker.detectForVideo(video.elt, now);
  
  if (result.landmarks && result.landmarks.length > 0) {
    // Using the index finger tip (landmark index 8)
    const indexTip = result.landmarks[0][8];
    // Calculate x and y from normalized coordinates; mirror the x coordinate.
    const x = width - (indexTip.x * width);
    const y = indexTip.y * height;
    const current = createVector(x, y);
    
    if (lastPoint) {
      drawing.push({ from: lastPoint.copy(), to: current.copy() });
    }
    lastPoint = current;
  } else {
    lastPoint = null;
  }
}

function draw() {
  background(255);
  
  // Draw the mirrored video feed.
  push();
  translate(width, 0);
  scale(-1, 1);
  image(video, 0, 0, width, height);
  pop();
  
  // Draw the strokes; the positions have been mirrored in trackHand().
  stroke(0, 102, 255);
  strokeWeight(parseInt(brushSizeSlider.value));
  for (let segment of drawing) {
    line(segment.from.x, segment.from.y, segment.to.x, segment.to.y);
  }
}

function startRecording() {
  recordedChunks = [];
  const canvasStream = document.querySelector('canvas').captureStream(30);
  mediaRecorder = new MediaRecorder(canvasStream, {
    mimeType: "video/webm; codecs=vp9"
  });
  mediaRecorder.ondataavailable = function(e) {
    if (e.data.size > 0) recordedChunks.push(e.data);
  };
  mediaRecorder.onstop = function() {
    const blob = new Blob(recordedChunks, { type: "video/webm" });
    const url = URL.createObjectURL(blob);
    const link = document.getElementById('downloadLink');
    link.href = url;
    link.download = "gesture_drawing.webm";
    link.textContent = "Download Video";
  };
  mediaRecorder.start();
  recording = true;
}

function stopRecording(callback) {
  mediaRecorder.stop();
  recording = false;
  if (callback) callback();
}

// Expose p5.js functions globally
window.setup = setup;
window.draw = draw;
