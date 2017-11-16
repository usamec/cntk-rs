git#[macro_use]
extern crate cntk;
extern crate rand;

use cntk::{Variable, Function, Value, Learner, Trainer, DoubleParameterSchedule, DataMap, Axis};
use cntk::ParameterInitializer;
use cntk::ReplacementMap;
use cntk::Shape;
use cntk::ops::*;
use cntk::DeviceDescriptor;
use rand::distributions::{IndependentSample, Range, StudentT};

fn build_dataset() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = rand::thread_rng();
    let step_range = Range::new(-0.05f32, 0.05f32);
    let start_range = Range::new(-0.5f32, 0.5f32);
    let noise_dist = StudentT::new(1.0);
    let coin = Range::new(0, 2);

    let step_num_range = Range::new(10, 20);
    let num_samples = 10000;
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut sum = 0.;
    let mut sqr_sum = 0.;
    let mut count = 0.;
    for i in 0..num_samples {
        let mut output: Vec<f32> = Vec::new();

        let mut current = (start_range.ind_sample(&mut rng), start_range.ind_sample(&mut rng));
        let mut current_vel = (0.0, 0.0);
        let num_steps = step_num_range.ind_sample(&mut rng);
        for j in 0..num_steps {
            output.push(current.0);
            output.push(current.1);
            count += 2.;
            sum += current.0;
            sum += current.1;
            sqr_sum += (current.0*current.0);
            sqr_sum += (current.1*current.1);
            current_vel.0 += step_range.ind_sample(&mut rng);
            current_vel.1 += step_range.ind_sample(&mut rng);
            if current_vel.0 < -1. {
                current_vel.0 = -1.;
            }
            if current_vel.0 > 1. {
                current_vel.0 = 1.;
            }
            if current_vel.1 < -1. {
                current_vel.1 = -1.;
            }
            if current_vel.1 > 1. {
                current_vel.1 = 1.;
            }
            current.0 += current_vel.0;
            current.1 += current_vel.1;
            /*if coin.ind_sample(&mut rng) == 0 {
                current.0 += step_range.ind_sample(&mut rng);
            } else {
                current.1 += step_range.ind_sample(&mut rng);
            }*/
        }

        let input = output.iter().map(|&x| x + 0.1 * noise_dist.ind_sample(&mut rng) as f32).collect::<Vec<_>>();

        inputs.push(input);
        outputs.push(output);
    }

    println!("total dataset variance {}", sqr_sum / count - (sum / count) * (sum / count));

    (inputs, outputs)
}

fn convolution_over_sequence<T: Into<Variable>>(input: T, input_size: usize, output_size: usize, width: usize) -> Function {
    let w = Variable::parameter(&Shape::new(&vec!(width, input_size, output_size)), &ParameterInitializer::constant(1.0), DeviceDescriptor::cpu());
    let b = Variable::parameter(&Shape::new(&vec!(1, output_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let inputv = input.into();

    let new_output = plus(convolution(w, transpose_axes(unpack(&inputv, 0.), &Axis::new(0), &Axis::new(1)), &Shape::new(vec!(1, input_size))), b);

    to_sequence_like(transpose_axes(new_output, &Axis::new(0), &Axis::new(1)), &inputv)
}

fn qrnn_layer<T: Into<Variable>>(input: T, input_size: usize, output_size: usize, width: usize, forward: bool) -> Function {
    let inputv = input.into();
    let one = Variable::constant_scalar(1.0);
    let new_values = tanh(convolution_over_sequence(&inputv, input_size, output_size, width));
    let gates = sigmoid(convolution_over_sequence(&inputv, input_size, output_size, width));

    let placeholder = Variable::placeholder(&Shape::new(vec!(output_size)));

    let output = plus(element_times(&gates, &new_values), element_times(minus(one, &gates), &placeholder));
    let placeholder_replacement = if (forward) {
        past_value(&output)
    } else {
        future_value(&output)
    };

    let replacements = replacementmap!{&placeholder => &placeholder_replacement};

    output.replace_placeholders(&replacements)
}

fn bidirectional_qrnn_layer<T: Into<Variable>>(input: T, input_size: usize, output_size: usize, width: usize) -> Function {
    let inputv = input.into();
    let forward = qrnn_layer(&inputv, input_size, output_size, width, true);
    let backward = qrnn_layer(&inputv, input_size, output_size, width, false);

    let combined = splice(&vec!(&Variable::from(forward), &Variable::from(backward)), &Axis::new(0));
    combined
}

fn linear_layer<T: Into<Variable>>(input: T, input_size: usize, output_size: usize) -> Function {
    let w = Variable::parameter(&Shape::new(&vec!(output_size, input_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let b = Variable::parameter(&Shape::new(&vec!(output_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    return plus(&b, times(&w, input));
}

fn main() {
    let (input_data, output_data) = build_dataset();

    let x = Variable::input_variable(&Shape::new(&vec!(2)));
    let y = Variable::input_variable(&Shape::new(&vec!(2)));

    let hidden_1 = bidirectional_qrnn_layer(&x, 2, 10, 3);
    let hidden_2 = bidirectional_qrnn_layer(&hidden_1, 20, 10, 3);
    let output = linear_layer(&hidden_2, 20, 2);

    let loss = reduce_mean(squared_error(&output, &y), &Axis::all());

    let all_parameters = output.parameters();

    let learner = Learner::sgd(&all_parameters, &DoubleParameterSchedule::constant(0.01));
    let trainer = Trainer::new(&output, &loss, &learner);

    let batch_size = 10;


    println!("training start");
    for iter in 0..50 {
        let mut total_loss = 0.0;


        for batch_num in 0..1000 {
            let value = Value::batch_of_sequences(&x.shape(), &input_data[batch_num*batch_size..(batch_num+1)*batch_size], DeviceDescriptor::cpu());
            let ovalue = Value::batch_of_sequences(&y.shape(), &output_data[batch_num*batch_size..(batch_num+1)*batch_size], DeviceDescriptor::cpu());
            let datamap = datamap!{&x => &value, &y => &ovalue};
            let mut outdatamap = outdatamap!{&output, &loss};

            trainer.train_minibatch(&datamap, &mut outdatamap, DeviceDescriptor::cpu());
            let output_val = outdatamap.get(&output).unwrap().to_vec();
            let loss_val = outdatamap.get(&loss).unwrap().to_vec();

            total_loss += loss_val[0];
        }
        println!("loss {:?}", total_loss / 2000.0);
    }
    println!("training end");
}
