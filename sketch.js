let video;
let model;
let predictions = [];
var capture;
let frame = 0;
let count = 0;
let brain;
let end = 0;

function preload() {
  // load in the graphic and sound assets of the app

  hand1 = loadAnimation('images/hand1/lights-01.png', 'images/hand1/lights-02.png');
  hand1.playing = false;

  hand2 = loadAnimation('images/hand2/lights-03.png');
  hand2.playing = false;

  hand3 = loadAnimation('images/hand3/lights-04.png');
  hand3.playing = false;

  hand4 = loadAnimation('images/hand4/lights-05.png');
  hand4.playing = false;

  hand5 = loadAnimation('images/hand5/lights-06.png');
  hand5.playing = false;

  hand6 = loadAnimation('images/hand6/lights-07.png');
  hand6.playing = false;

  hand7 = loadAnimation('images/hand7/lights-08.png', 'images/hand7/lights-09.png');
  hand7.playing = false;

  video = createCapture(VIDEO, () => {
    loadHandTrackingModel();
  });
  video.hide();

  song = loadSound('images/magicsound.mp3');
}

function setup() {
  createCanvas(640, 480);

  // Scale and position the handwashing icons

  spr7 = createSprite(width / 2, height / 2);
  spr7.addAnimation("default", hand7);
  spr7.scale = 0.25;
  spr7.position.x = width / 2;
  spr7.position.y = height / 2;

  spr6 = createSprite(width / 2, height / 2);
  spr6.addAnimation("default", hand6);
  spr6.scale = 0.25;
  spr6.position.x = width / 2;
  spr6.position.y = height / 2;

  spr5 = createSprite(width / 2, height / 2);
  spr5.addAnimation("default", hand5);
  spr5.scale = 0.25;
  spr5.position.x = width / 2;
  spr5.position.y = height / 2;

  spr4 = createSprite(width / 2, height / 2);
  spr4.addAnimation("default", hand4);
  spr4.scale = 0.25;
  spr4.position.x = width / 2;
  spr4.position.y = height / 2;

  spr3 = createSprite(width / 2, height / 2);
  spr3.addAnimation("default", hand3);
  spr3.scale = 0.25;
  spr3.position.x = width / 2;
  spr3.position.y = height / 2;

  spr2 = createSprite(width / 2, height / 2);
  spr2.addAnimation("default", hand2);
  spr2.scale = 0.25;
  spr2.position.x = width / 2;
  spr2.position.y = height / 2;

  spr1 = createSprite(width / 2, height / 2);
  spr1.addAnimation("default", hand1);
  spr1.scale = 0.25;
  spr1.position.x = width / 2;
  spr1.position.y = height / 2;

}

async function loadHandTrackingModel() {
  // Load the MediaPipe handpose model.
  model = await handpose.load();
  predictHand();
  let options = {
    inputs: 63,
    outputs: 7,
    task: 'classification',
    debug: true
  }
  // Load and read the handson trained model/neural network
  const modelInfo = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin',
  };
  let modelJson = await fetch(modelInfo.model);
  modelJson = await modelJson.text();
  const modelJsonFile = new File([modelJson], 'model.json', {
    type: 'application/json'
  });
  let weightsBlob = await fetch(modelInfo.weights);
  weightsBlob = await weightsBlob.blob();
  const weightsBlobFile = new File([weightsBlob], 'model.weights.bin', {
    type: 'application/macbinary',
  });

  brain = await tf.loadLayersModel(tf.io.browserFiles([modelJsonFile, weightsBlobFile]));
  brainLoaded();
}

function brainLoaded() {
  console.log('Hand pose classification ready!');
  classifyPose();
}

async function classifyPose() {
  if (predictions.length > 0) {
    const landmarks = predictions[0].landmarks;
    let inputs = [];
    for (let i = 0; i < landmarks.length; i++) {
      inputs.push(landmarks[i][0] / 640);
      inputs.push(landmarks[i][1] / 480);
      inputs.push((landmarks[i][2] + 80) / 80);
    }
    const output = tf.tidy(() => {
      return brain.predict(tf.tensor(inputs, [1, 63]));
    });
    const result = await output.array();
    gotResult(result);
  } else {
    setTimeout(classifyPose, 100);
  }
}

function gotResult(results) {
  console.log('1 = ' + results[0][0]);
  console.log('2 = ' + results[0][1]);
  console.log('3 = ' + results[0][2]);
  console.log('4 = ' + results[0][3]);
  console.log('5 = ' + results[0][4]);
  console.log('6 = ' + results[0][5]);
  console.log('7 = ' + results[0][6]);

  if (results[0] && results[0][0]) {
    if (results[0][0] > 0.0001) {
      hand1();
    }
    if (results[0][1] > 0.001) {
      hand2();
    }
    if (results[0][2] > 0.5) {
      hand3();
    }
    if (results[0][3] > 0.0001) {
      hand4();
    }
    if (results[0][4] > 0.98) {
      hand5();
    }
    if (results[0][5] > 0.0001) {
      hand6();
    }
    if (results[0][6] > 0.0001) {
      hand7();
    }
  }

  function hand1() {
    if (count % 100 == 0) {
      if (spr1.animation.getFrame() == 1 && spr1.removed == false) {
        spr1.remove();
        song.play();
        count++;
      } else if (spr1.animation.getFrame() == 0) {
        spr1.animation.nextFrame();
        count++;
      }
    }
  }

  function hand2() {
    if (count % 100 == 0 && spr1.removed) {
      if (spr2.animation.getFrame() == 0 && spr2.removed == false) {
        spr2.remove();
        song.play();
        count++
      }
    }
  }

  function hand3() {
    if (count % 100 == 0 && spr2.removed) {
      if (spr3.animation.getFrame() == 0 && spr3.removed == false) {
        spr3.remove();
        song.play();
        count++
      }
    }
  }

  function hand4() {
    if (count % 100 == 0 && spr3.removed) {
      if (spr4.animation.getFrame() == 0 && spr4.removed == false) {
        spr4.remove();
        song.play();
        count++
      }
    }
  }

  function hand5() {
    if (count % 100 == 0 && spr4.removed) {
      if (spr5.animation.getFrame() == 0 && spr5.removed == false) {
        spr5.remove();
        song.play();
        count++
      }
    }
  }

  function hand6() {
    if (count % 100 == 0 && spr5.removed) {
      if (spr6.animation.getFrame() == 0 && spr6.removed == false) {
        spr6.remove();
        song.play();
        count++
      }
    }
  }

  function hand7() {
    if (count % 100 == 0 && spr6.removed) {
      if (spr7.animation.getFrame() == 0) {
        spr7.animation.nextFrame();
      } else {
        song.play();
        setTimeout(setup);
      }
    }
  }

  classifyPose();
}

function modelLoaded() {
  console.log('Hand pose model ready');
}

function draw() {
  background(255, 209, 69);
  if (model) {
    image(video, 0, 0);
  }
  if (predictions.length > 0) {
    // We can call both functions to draw all keypoints and the skeletons
    drawKeypoints();
    drawSkeleton();
    count++;
  }
  drawSprites();
}

async function predictHand() {
  // Pass in a video stream (or an image, canvas, or 3D tensor) to obtain a
  // hand prediction from the MediaPipe graph.
  predictions = await model.estimateHands(video.elt);
  setTimeout(() => predictHand(), 0);
}

//function that draws out bacteria
function drawKeypoints() {
  let prediction = predictions[0];
  for (let j = 0; j < prediction.landmarks.length; j++) {
    let keypoint = prediction.landmarks[j];
    fill(235, 68, 59);
    noStroke();
    ellipse(keypoint[0], keypoint[1], 10, 10);
  }
}

//function that estimates the skeletal structure of the hand from handpose
function drawSkeleton() {
  stroke(235, 68, 59);
  let annotations = predictions[0].annotations;
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