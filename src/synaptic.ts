import { Neuron, Layer, Network, Trainer } from 'synaptic';

import * as mnist from 'mnist';
let data = mnist.set(2000, 8000);
console.log(data.training.length);
console.log(data.test.length);
let res;

let inputLayer: Layer = new Layer(784);
let hiddenLayer1: Layer = new Layer(3);
let outputLayer: Layer = new Layer(10);

inputLayer.set({
    squash: Neuron.squash.IDENTITY,
    bias: 0.1
});

hiddenLayer1.set({
    squash: Neuron.squash.ReLU,
    bias: 0.1
});

inputLayer.project(hiddenLayer1);
hiddenLayer1.project(outputLayer);

var learningRate = .4;

var myNetwork = new Network({
    input: inputLayer,
    hidden: [hiddenLayer1],
    output: outputLayer
});

myNetwork.optimize();
let trainer = new Trainer(myNetwork);

console.time('train');
trainer.train(data.training, {
    rate: (iterations, error) => {
        return Math.max(1 / iterations, 0.15);
    },
    cost: Trainer.cost.MSE,
    shuffle: true,
    error: .05,
    iterations: 20000,
    log: 1
});
console.timeEnd('train');

let acc_count = 0;
res = trainer.test(data.test);
console.log(`Error: ${res.error * 100}%`);
// console.log(myNetwork.toJSON());