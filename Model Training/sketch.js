// Handpose Classification
let brain;

function setup() {
  createCanvas(640, 480);
  let options = {
    inputs: 63,
    outputs: 7,
    task: 'classification',
    debug: true
  }
  brain = ml5.neuralNetwork(options);
  // Load in the handson.json file into the neural network
  brain.loadData('handson.json', dataReady);
}

function dataReady() {
  brain.normalizeData();
  //train the neural network 1000 times with the default batchsize as 32
  brain.train({
    epochs: 1000
  }, finished);
}

function finished() {
  console.log('model trained');
  //saving the trained model for use
  brain.save();
}