import * as convnetjs from 'convnetjs';
import * as mnist from 'mnist';
let data = mnist.set(2000, 8000);
console.log(data.training.length);
console.log(data.test.length);

let getMax = (arr: number[]) => {
    return arr.indexOf(arr.reduce((prev, curr) => {
        return Math.max(prev, curr);
    }));
}

let layer_defs = [];
// layer_defs.push({type:'input', out_sx:28, out_sy:28, out_depth:1});
// layer_defs.push({type:'fc', num_neurons: 3, activation: 'relu'});
// layer_defs.push({type:'fc', num_neurons: 10});
layer_defs.push({type:'input', out_sx:24, out_sy:24, out_depth:1});
layer_defs.push({type:'conv', sx:5, filters:8, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx:2, stride:2});
layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx:3, stride:3});
layer_defs.push({type:'softmax', num_classes:10});
let net = new convnetjs.Net();
net.makeLayers(layer_defs);
// let trainer = new convnetjs.Trainer(net, { learning_rate: 0.01});
// let trainer = new convnetjs.Trainer(net, {method:'adam', batch_size:20, l2_decay:0.001});
// var trainer = new convnetjs.Trainer(net, {
//     method: 'adagrad',
//     l2_decay: 0.001, 
//     l1_decay: 0.001,
//     batch_size: 10
// });
 
let trainer = new convnetjs.Trainer(net, {method:'adadelta', batch_size:20, l2_decay:0.001});
// if your loss on top is
let stats;

let accurate = 0
let iterations = 20;
for (let iteration = 1; iteration <= iterations; iteration++) {
    data.training.forEach(item => {
        stats = trainer.train(new convnetjs.Vol(item.input), new convnetjs.Vol(item.output));
    });
    console.log(iteration, stats);
}
data.test.forEach(item => {
    stats = net.forward(new convnetjs.Vol(item.input));
    accurate += getMax(stats.w) === item.output.indexOf(1) ? 1 : 0;
});
console.log(stats, accurate / data.test.length);
// let stats = trainer.train(x, results[0]);
// stats = net.forward(x);

// let stats = net.forward(x);
// console.log(stats);
// console.log(stats);
// console.log('training...');
// console.time('training');
// for (let i = 0; i < 20; i++) {
//     let stats = trainer.train(x, results[i]);
//     console.log(stats);
// }
// console.timeEnd('training');

// let input = new convnetjs.Vol([data.test[0].input]);
// let res = trainer.net.forward(input);
// console.log(res);
// console.log(res.w.indexOf((<number[]> res.w).reduce((curr, item) => {
//     return Math.max(curr, item);
// }, 0)));

// let res = net.forward(20)
// console.log(res);
