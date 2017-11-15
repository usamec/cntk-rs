#[macro_use]
extern crate cntk;
extern crate regex;
extern crate rand;

use cntk::{Variable, Function, Value, Learner, Trainer, DoubleParameterSchedule, DataMap, Axis};
use cntk::ParameterInitializer;
use cntk::set_max_num_cpu_threads;
use cntk::Shape;
use cntk::ops::*;
use cntk::DeviceDescriptor;
use cntk::ReplacementMap;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};

fn tokenize(s: &str) -> Vec<String> {
    let re = Regex::new(r"\b\w\w+\b").unwrap();
    re.captures_iter(s).map(|x| x[0].to_lowercase()).collect()
}

fn build_vocab(tokens: &[String]) -> (HashMap<String, usize>, Vec<usize>) {
    let counts = tokens.iter().fold(HashMap::new(), |mut map, x| {
        if (!map.contains_key(x)) {
            map.insert(x.clone(), 0);
        }
        *map.get_mut(x).unwrap() += 1;
        map
    });
    tokens.iter().fold((HashMap::new(), Vec::new()), |(mut map, mut data), x| {
        if counts[x] < 10 {
            data.push(0)
        } else {
            if (!map.contains_key(x)) {
                let id = map.len() + 1;
                map.insert(x.clone(), id);
            }
            data.push(map[x]);
        }
        (map, data)
    })
}

fn linear_layer<T: Into<Variable>>(input: T, input_size: usize, output_size: usize) -> Function {
    let w = Variable::parameter(&Shape::new(&vec!(output_size, input_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let b = Variable::parameter(&Shape::new(&vec!(output_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    return plus(&b, times(&w, input));
}

fn gru_layer<T: Into<Variable>>(input: T, input_size: usize, hidden_size: usize, init_value: Option<&Variable>) -> Function {
    let inputv = input.into();

    let placeholder = Variable::placeholder(&Shape::new(vec!(hidden_size)));

    let one = Variable::constant_scalar(1.0);
    let wou = Variable::parameter(&Shape::new(vec!(hidden_size, hidden_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let wiu = Variable::parameter(&Shape::new(vec!(hidden_size, input_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let bu = Variable::parameter(&Shape::new(vec!(hidden_size)), &ParameterInitializer::constant(-1.0), DeviceDescriptor::cpu());
    let update_gate = sigmoid(plus(plus(times(&wou, &placeholder), times(&wiu, &inputv)), &bu));

    let wor = Variable::parameter(&Shape::new(vec!(hidden_size, hidden_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let wir = Variable::parameter(&Shape::new(vec!(hidden_size, input_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let br = Variable::parameter(&Shape::new(vec!(hidden_size)), &ParameterInitializer::constant(1.0), DeviceDescriptor::cpu());
    let reset_gate = sigmoid(plus(plus(times(&wor, &placeholder), times(&wir, &inputv)), &br));

    let woo = Variable::parameter(&Shape::new(vec!(hidden_size, hidden_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let wio = Variable::parameter(&Shape::new(vec!(hidden_size, input_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let bo = Variable::parameter(&Shape::new(vec!(hidden_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let new_value = tanh(plus(plus(times(&wio, &inputv), &bo), element_times(&reset_gate, times(&woo, &placeholder))));

    let output = plus(element_times(&update_gate, new_value), element_times(minus(one, &update_gate), &placeholder));

    let placeholder_replacement = match init_value {
        Some(var) => past_value_with_init(&output, broadcast_as(var, &inputv)),
        None => past_value(&output)
    };

    let replacements = replacementmap!{&placeholder => &placeholder_replacement};

    output.replace_placeholders(&replacements)
}

fn main() {
    set_max_num_cpu_threads(1);
    let file = File::open("data/shakespeare_input.txt").unwrap();
    let mut buf_reader = BufReader::new(file);
    let mut contents = String::new();
    buf_reader.read_to_string(&mut contents).unwrap();
    let tokens = tokenize(&contents);

    let (vocab, translated_tokens) = build_vocab(&tokens);

    let num_words = vocab.len();
    let embedding_size = 100;
    
    let num_tokens = num_words + 1; // num_words tokens has special meaning (end of sequence)

    let input_axis = Axis::named_dynamic("input");
    let label_axis = Axis::named_dynamic("label");
    let encoder_input = Variable::create(&Shape::new(vec!(num_tokens)), true, false, "X", &vec!(&input_axis, &Axis::default_batch_axis()));
    let decoder_input = Variable::create(&Shape::new(vec!(num_tokens)), true, false, "X2", &vec!(&label_axis, &Axis::default_batch_axis()));
    let decoder_labels = Variable::create(&Shape::new(vec!(num_tokens)), true, false, "Y", &vec!(&label_axis, &Axis::default_batch_axis()));

    let layer1_encoder = gru_layer(&encoder_input, num_tokens, 100, None);
    let layer2_encoder = gru_layer(&layer1_encoder, 100, 100, None);

    let layer1_encoder_last = last(&layer1_encoder);
    let layer2_encoder_last = last(&layer2_encoder);

    let layer1_decoder = gru_layer(&decoder_input, num_tokens, 100, Some(&Variable::from(layer1_encoder_last)));
    let layer2_decoder = gru_layer(&layer1_decoder, 100, 100, Some(&Variable::from(layer2_encoder_last)));

    let decoder_output_weights = Variable::parameter(&Shape::new(vec!(100, num_tokens)), &ParameterInitializer::constant(0.0), DeviceDescriptor::cpu());
    let decoder_output_biases = Variable::parameter(&Shape::new(vec!(1, num_tokens)), &ParameterInitializer::constant(0.0), DeviceDescriptor::cpu());

    let decoder_logits = plus(transpose_times(&decoder_output_weights, &layer2_decoder),
                             reshape(&decoder_output_biases, &Shape::new(vec!(num_tokens))));
    let decoder_probs = softmax(&decoder_logits);
    let loss = reduce_mean(cross_entropy_with_softmax(&decoder_logits, &decoder_labels), &Axis::all());
    let decoder_predictions = argmax(&decoder_probs, &Axis::new(0));

    let all_parameters = loss.parameters();
    let learner = Learner::momentum_sgd(&all_parameters, &DoubleParameterSchedule::constant(0.1), &DoubleParameterSchedule::constant(0.95));
    let trainer = Trainer::new(&layer2_decoder, &loss, &learner);

    let mut loss_sum = 0.0;
    let batch_size = 5;
    let mut rng = rand::thread_rng();
    let start_range = Range::new(0, translated_tokens.len() - 10);
    let size_range = Range::new(3, 7);
    for iter in 0..1000000 {
        let mut encoder_input_batch = Vec::new();
        let mut decoder_input_batch = Vec::new();
        let mut decoder_labels_batch = Vec::new();
        for i in 0..batch_size {
            let pos = start_range.ind_sample(&mut rng);
            let len = size_range.ind_sample(&mut rng);
            let encoder_input_sample = translated_tokens[pos..pos + len].to_owned();
            let mut decoder_input_sample = vec!(num_words);
            decoder_input_sample.extend(encoder_input_sample.iter().rev());
            let mut decoder_labels_sample = vec!();
            decoder_labels_sample.extend(encoder_input_sample.iter().rev());
            decoder_labels_sample.push(num_words);
            encoder_input_batch.push(encoder_input_sample);
            decoder_input_batch.push(decoder_input_sample);
            decoder_labels_batch.push(decoder_labels_sample);
        }
        let encoder_input_value = Value::batch_of_one_hot_sequences(&Shape::new(vec!(num_tokens)), &encoder_input_batch, DeviceDescriptor::cpu());
        let decoder_input_value = Value::batch_of_one_hot_sequences(&Shape::new(vec!(num_tokens)), &decoder_input_batch, DeviceDescriptor::cpu());
        let decoder_labels_value = Value::batch_of_one_hot_sequences(&Shape::new(vec!(num_tokens)), &decoder_labels_batch, DeviceDescriptor::cpu());

        let datamap = datamap!{&encoder_input => &encoder_input_value,
                               &decoder_input => &decoder_input_value,
                               &decoder_labels => &decoder_labels_value};
        let mut outdatamap = outdatamap!{&loss};
        trainer.train_minibatch(&datamap, &mut outdatamap, DeviceDescriptor::cpu());
        let loss_val = outdatamap.get(&loss).unwrap().to_vec();

        loss_sum += loss_val[0];
        if (iter+1) % 100 == 0 {
            println!("iter {} loss val {:?}", iter, loss_sum / 1002.0);

            let datamap = datamap!{&encoder_input => &encoder_input_value,
                                   &decoder_input => &decoder_input_value};
            let mut outdatamap = outdatamap!{&decoder_probs, &decoder_predictions};

            decoder_predictions.evaluate(&datamap, &mut outdatamap, DeviceDescriptor::cpu());

            println!("excepted  {:?}", decoder_labels_batch[0]);
            let preds = outdatamap.get(&decoder_predictions).unwrap().to_vec();
            println!("predicted {:?}", &preds[..decoder_labels_batch[0].len()]);
            println!("");

            loss_sum = 0.0;
        }
    }
}