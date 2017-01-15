import * as convnet from 'convnetjs';
import * as mnist from 'mnist';

let layers = [];

layers.push({
    type: 'input', out_sx: 1,
    out_sy: 1,
    out_depth: 2
});
// layers.push({
//     type: 'svm',
//     num_classes: 2
// });

layers.push({
    type: 'fc',
    num_neurons: 3,
    activation: 'relu'
})

let net = new convnet.Net();
net.makeLayers(layers);

let x = new convnet.Vol(1, 1, 200);
x.w[0] = 0.5;
x.w[1] = -1.3;

// for (let i = 0; i < 100000; i++) {
//     let val = net.forward(x, 20);
//     net.backward();
//     console.log(val);
// }

var trainer = new convnet.Trainer(net, {method: 'adadelta', l2_decay: 0.001,
                                    batch_size: 10});

let stats = trainer.train(x, 3);
console.log(stats);

// let stats = trainer.train(x, 3);
// console.log(x);

// let set = mnist.set(10, 20);
// let trainingSet = set.training;
// let testSet = set.test;

// let x = new convnet.Vol(1, 1, 1);
// x.w[0] = 1;

// layers.push({
//     type: 'input',
//     out_sx: 1,
//     out_sy: 1,
//     out_depth: 1
// });

// layers.push({
//     type: 'conv',
//     sx: 1,
//     filters: 1,
//     stride: 1,
//     activation: 'relu'
// });

// layers.push({
//     type: 'softmax',
//     num_classes: 1
// });

// let net = new convnet.Net();
// net.makeLayers(layers);

// let trainer = new convnet.SGDTrainer(net, {
//     method: 'adadelta',
//     batch_size: 20,
//     l2_decay: 0.001
// });

// let stats = trainer.train(x, 3);
// console.log(stats);


// trainer.train(trainingSet.map(data => {
//     return data.input;
// }));
// console.log(trainer);