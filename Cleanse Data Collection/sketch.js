// Hand pose classification
// This code is based on
// ml5.js: Pose Classification

let video;
let model;
let predictions = [];

let brain;

let state = 'waiting';
let targeLabel;

function preload() {
  video = createCapture(VIDEO, () => {
    loadHandTrackingModel();
  });
  video.hide();
}

function keyPressed() {
  if (key == 's') {
    brain.saveData();
  } else {
    //press 0-6 to train the 7 different handwashing techniques respectively
    targetLabel = key;
    console.log(targetLabel);
    setTimeout(function() {
      console.log('collecting');
      state = 'collecting';
      setTimeout(function() {
        console.log('not collecting');
        state = 'waiting';
      }, 10000);
    }, 100);
  }
}

async function loadHandTrackingModel() {
  // Load the MediaPipe handpose model.
  model = await handpose.load();
  predictHand();
}

async function predictHand() {
  predictions = await model.estimateHands(video.elt);
  if (predictions.length > 0) {
    const landmarks = predictions[0].landmarks;
    if (state == 'collecting') {
      let inputs = [];
      for (let i = 0; i < landmarks.length; i++) {
        inputs.push(landmarks[i][0]);
        inputs.push(landmarks[i][1]);
        inputs.push(landmarks[i][2]);
      }
      let target = [targetLabel];
      brain.addData(inputs, target);
    }
  }

  setTimeout(() => predictHand(), 100);
}

function setup() {
  createCanvas(640, 480);

  let options = {
    inputs: 63,
    outputs: 7,
    task: 'classification',
    debug: true
  }
  //creating our own neural network
  brain = ml5.neuralNetwork(options);
}


function modelLoaded() {
  console.log('hand pose ready');
}

function draw() {
  if (model) image(video, 0, 0);
  if (predictions.length > 0) {
    // We can call both functions to draw all keypoints and the skeletons for the visualisation of the training
    drawKeypoints();
    drawSkeleton();
  }
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints() {
  let prediction = predictions[0];
  for (let j = 0; j < prediction.landmarks.length; j++) {
    let keypoint = prediction.landmarks[j];
    fill(255, 0, 0);
    noStroke();
    ellipse(keypoint[0], keypoint[1], 10, 10);
  }
}

// A function to draw the skeletons
function drawSkeleton() {
  let annotations = predictions[0].annotations;
  stroke(255, 0, 0);
  for (let j = 0; j < annotations.thumb.length - 1; j++) {
    line(annotations.thumb[j][0], annotations.thumb[j][1], annotations.thumb[j + 1][0], annotations.thumb[j + 1][1]);
  }
  for (let j = 0; j < annotations.indexFinger.length - 1; j++) {
    line(annotations.indexFinger[j][0], annotations.indexFinger[j][1], annotations.indexFinger[j + 1][0], annotations.indexFinger[j + 1][1]);
  }
  for (let j = 0; j < annotations.middleFinger.length - 1; j++) {
    line(annotations.middleFinger[j][0], annotations.middleFinger[j][1], annotations.middleFinger[j + 1][0], annotations.middleFinger[j + 1][1]);
  }
  for (let j = 0; j < annotations.ringFinger.length - 1; j++) {
    line(annotations.ringFinger[j][0], annotations.ringFinger[j][1], annotations.ringFinger[j + 1][0], annotations.ringFinger[j + 1][1]);
  }
  for (let j = 0; j < annotations.pinky.length - 1; j++) {
    line(annotations.pinky[j][0], annotations.pinky[j][1], annotations.pinky[j + 1][0], annotations.pinky[j + 1][1]);
  }

  line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.thumb[0][0], annotations.thumb[0][1]);
  line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.indexFinger[0][0], annotations.indexFinger[0][1]);
  line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.middleFinger[0][0], annotations.middleFinger[0][1]);
  line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.ringFinger[0][0], annotations.ringFinger[0][1]);
  line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.pinky[0][0], annotations.pinky[0][1]);

}