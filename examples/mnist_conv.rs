extern crate cntk;
extern crate mnist;

use cntk::{Variable, Function, Value, Learner, Trainer, DoubleParameterSchedule, DataMap, Axis};
use cntk::ParameterInitializer;
use cntk::Shape;
use cntk::ops::*;
use cntk::DeviceDescriptor;

use mnist::{Mnist, MnistBuilder};

fn linear_layer(input: &Variable, input_size: usize, output_size: usize) -> Variable {
    let w = Variable::parameter(&Shape::from_slice(&vec!(output_size, input_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let b = Variable::parameter(&Shape::from_slice(&vec!(output_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    return plus(&b, &times(&w, input));
}

fn mlp_layer(input: &Variable, input_size: usize, output_size: usize) -> Variable {
    return tanh(&linear_layer(input, input_size, output_size));
}

fn conv_layer(input: &Variable, input_channels: usize, output_channels: usize, filter_size: usize) -> Variable {
    let w = Variable::parameter(&Shape::from_slice(&vec!(filter_size, filter_size, input_channels, output_channels)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let b = Variable::parameter(&Shape::from_slice(&vec!(1, 1, output_channels)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    return relu(&plus(&b, &convolution(&w, input, &Shape::from_slice(&vec!(1, 1, input_channels)))));
}

fn pooling_layer(input: &Variable, pool_size: usize) -> Variable {
    return max_pooling(input, &Shape::from_slice(&vec!(pool_size, pool_size)), &Shape::from_slice(&vec!(pool_size, pool_size)));
}

fn main() {
    let x = Variable::input_variable(&Shape::from_slice(&vec!(28,28,1)));
    let y = Variable::input_variable(&Shape::from_slice(&vec!(10)));
    let h1 = conv_layer(&x, 1, 10, 3);
    let h2 = pooling_layer(&h1, 2);
    let h3 = conv_layer(&h2, 10, 10, 3);
    let h4 = pooling_layer(&h3, 2);
    let h5 = mlp_layer(&reshape(&h4, &Shape::from_slice(&vec!(7*7*10))), 7*7*10, 50);
    let output = linear_layer(&h5, 50, 10);
    let prediction = argmax(&output, &Axis::new(0));
    let loss = reduce_sum_all(&cross_entropy_with_softmax(&output, &y));
    let error_count = reduce_sum_all(&classification_error(&output, &y));

    let output_func = Function::from_variable(&output);
    let prediction_func = Function::from_variable(&prediction);
    let loss_func = Function::from_variable(&loss);
    let error_count_func = Function::from_variable(&error_count);

    let all_parameters = output_func.parameters();

    let learner = Learner::sgd(&all_parameters.iter().collect::<Vec<&Variable>>(), &DoubleParameterSchedule::constant(0.01));
    let trainer = Trainer::new_with_evalatuion(&output_func, &loss_func, &error_count_func, &learner);

    let (trn_size, rows, cols) = (50_000, 28, 28);

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
    for iter in 0..5 {
        let mut total_loss = 0.0;
        let mut total_error_count = 0.0;
        for batch_num in 0..1000 {
            let value = Value::batch(&x.shape(), &images[batch_num*batch_size*28*28..(batch_num+1)*batch_size*28*28], DeviceDescriptor::cpu());
            let ovalue = Value::batch(&y.shape(), &labels[batch_num*batch_size*10..(batch_num+1)*batch_size*10], DeviceDescriptor::cpu());
            let mut datamap = DataMap::new();
            datamap.add(&x, &value);
            datamap.add(&y, &ovalue);
            let mut outdatamap = DataMap::new();
            outdatamap.add_null(&output);
            outdatamap.add_null(&loss);
            outdatamap.add_null(&error_count);

            trainer.train_minibatch(&datamap, &mut outdatamap, DeviceDescriptor::cpu());
            let output_val = outdatamap.get(&output).unwrap().to_vec();
            let loss_val = outdatamap.get(&loss).unwrap().to_vec();
            let error_count_val = outdatamap.get(&error_count).unwrap().to_vec();

            total_loss += loss_val[0];
            total_error_count += error_count_val[0];
        }
        println!("loss {:?} error_count {}", total_loss / 1000.0, total_error_count / 50000.0);
    }
    println!("training end");

    let mut val_labels = vec!(0.0; 10_000*10);
    let val_images = val_img.iter().map(|&x| (x as f32) / 256.0).collect::<Vec<f32>>();

    for i in 0..val_lbl.len() {
        val_labels[i*10+val_lbl[i] as usize] = 1.0
    }

    let value = Value::batch(&x.shape(), &val_images, DeviceDescriptor::cpu());
    let mut datamap = DataMap::new();
    datamap.add(&x, &value);
    let mut outdatamap = DataMap::new();
    outdatamap.add_null(&prediction);

    prediction_func.evaluate(&datamap, &mut outdatamap, DeviceDescriptor::cpu());
    let result = outdatamap.get(&prediction).unwrap().to_vec();

    println!("error cnt {}/{}", result.iter().zip(val_lbl.iter()).map(|(&r, &l)| r as i32 != l as i32).fold(0, |sum, val| sum + val as i32), result.len());
}