#[macro_use]
extern crate cntk;
extern crate mnist;

use cntk::{Variable, Function, Value, Learner, Trainer, DoubleParameterSchedule, DataMap, Axis};
use cntk::ParameterInitializer;
use cntk::Shape;
use cntk::ops::*;
use cntk::DeviceDescriptor;

use mnist::{Mnist, MnistBuilder};

fn linear_layer<T: Into<Variable>>(input: T, input_size: usize, output_size: usize) -> Function {
    let w = Variable::parameter(&Shape::new(&vec!(output_size, input_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let b = Variable::parameter(&Shape::new(&vec!(output_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    return plus(&b, times(&w, input));
}

fn mlp_layer<T: Into<Variable>>(input: T, input_size: usize, output_size: usize) -> Function {
    return tanh(linear_layer(input, input_size, output_size));
}

fn main() {
    // Graph definition starts here
    // First we define symbolic variables. x is for input, y is for expected labels
    let x = Variable::input_variable(&Shape::new(&vec!(28*28)));
    let y = Variable::input_variable(&Shape::new(&vec!(10)));
    // We build first hidden layer. Its input is input variable.
    let h1 = mlp_layer(&x, 28*28, 200);
    // First layer serves as an input for second hidden layer
    let h2 = mlp_layer(h1, 200, 200);
    // We build softmax layer on top of second hidden layer
    let output = linear_layer(h2, 200, 10);
    let prediction = argmax(&output, &Axis::new(0));
    // Here is define the loss, which we are going to optimize
    let loss = reduce_sum(&cross_entropy_with_softmax(&output, &y), &Axis::all());
    let error_count = reduce_sum(&classification_error(&output, &y), &Axis::all());

    let all_parameters = output.parameters();

    // Here we define Learner and Trainer duo. Learner defines how are the parameters updated after each iteration.
    // Trainer oversees training.
    let learner = Learner::sgd(&all_parameters, &DoubleParameterSchedule::constant(0.01));
    let trainer = Trainer::new_with_evalatuion(&output, &loss, &error_count, &learner);


    // Loading and proprocessing of MNIST dataset
    let (trn_size, _rows, _cols) = (50_000, 28, 28);
    // Deconstruct the returned Mnist struct.
    let Mnist { trn_img, trn_lbl, val_img, val_lbl, .. } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let mut labels = vec!(0.0; 50_000*10);
    let images = trn_img.iter().map(|&x| (x as f32) / 256.0).collect::<Vec<f32>>();

    for i in 0..trn_lbl.len() {
        labels[i*10+trn_lbl[i] as usize] = 1.0
    }

    let batch_size = 50;

    println!("training start");
    for _iter in 0..5 {
        let mut total_loss = 0.0;
        let mut total_error_count = 0.0;
        for batch_num in 0..1000 {
            // Values contain real data
            let value = Value::batch_from_vec(&x.shape(), &images[batch_num*batch_size*28*28..(batch_num+1)*batch_size*28*28], DeviceDescriptor::cpu());
            let ovalue = Value::batch_from_vec(&y.shape(), &labels[batch_num*batch_size*10..(batch_num+1)*batch_size*10], DeviceDescriptor::cpu());

            // Here we bind values to input variables
            let datamap = datamap!{&x => &value, &y => &ovalue};

            // And here we define what variables we expect in output
            let mut outdatamap = outdatamap!{&output, &loss, &error_count};

            // This function does actual training iteration
            trainer.train_minibatch(&datamap, &mut outdatamap, DeviceDescriptor::cpu());

            // We retrieve relevant outputs
            let _output_val = outdatamap.get(&output).unwrap().to_vec();
            let loss_val = outdatamap.get(&loss).unwrap().to_vec();
            let error_count_val = outdatamap.get(&error_count).unwrap().to_vec();

            total_loss += loss_val[0];
            total_error_count += error_count_val[0];
        }
        println!("loss {:?} error_count {}", total_loss / 1000.0, total_error_count / 50000.0);
    }
    println!("training end");

    // Preprocess data for validation
    let mut val_labels = vec!(0.0; 10_000*10);
    let val_images = val_img.iter().map(|&x| (x as f32) / 256.0).collect::<Vec<f32>>();

    for i in 0..val_lbl.len() {
        val_labels[i*10+val_lbl[i] as usize] = 1.0
    }

    // Create value for validation input
    let value = Value::batch_from_vec(&x.shape(), &val_images, DeviceDescriptor::cpu());

    // Non macro syntax for datamap initializations, just for an example
    let mut datamap = DataMap::new();
    datamap.add(&x, &value);
    let mut outdatamap = DataMap::new();
    outdatamap.add_null(&prediction);

    // Get predictions
    prediction.evaluate(&datamap, &mut outdatamap, DeviceDescriptor::cpu());
    let result = outdatamap.get(&prediction).unwrap().to_vec();

    println!("error cnt {}/{}", result.iter().zip(val_lbl.iter()).map(|(&r, &l)| r as i32 != l as i32).fold(0, |sum, val| sum + val as i32), result.len());
}