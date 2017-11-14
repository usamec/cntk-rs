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
    tokens.iter().fold((HashMap::new(), Vec::new()), |(mut map, mut data), x| {
        if (!map.contains_key(x)) {
            let id = map.len();
            map.insert(x.clone(), id);
        }
        data.push(map[x]);
        (map, data)
    })
}

fn linear_layer<T: Into<Variable>>(input: T, input_size: usize, output_size: usize) -> Function {
    let w = Variable::parameter(&Shape::new(&vec!(output_size, input_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let b = Variable::parameter(&Shape::new(&vec!(output_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    return plus(&b, times(&w, input));
}

fn gru_layer<T: Into<Variable>>(input: T, input_size: usize, hidden_size: usize, init: Option<&Variable>) -> Function {
    let inputv = input.into();

    let placeholder = Variable::placeholder(&Shape::new(vec!(hidden_size)));

    let one = Variable::constant_scalar(1.0);
    let wou = Variable::parameter(&Shape::new(vec!(hidden_size, hidden_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let wiu = Variable::parameter(&Shape::new(vec!(hidden_size, input_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let bu = Variable::parameter(&Shape::new(vec!(hidden_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let update_gate = sigmoid(plus(plus(times(&wou, &placeholder), times(&wiu, &inputv)), &bu));

    let wor = Variable::parameter(&Shape::new(vec!(hidden_size, hidden_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let wir = Variable::parameter(&Shape::new(vec!(hidden_size, input_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let br = Variable::parameter(&Shape::new(vec!(hidden_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let reset_gate = sigmoid(plus(plus(times(&wor, &placeholder), times(&wir, &inputv)), &br));

    let woo = Variable::parameter(&Shape::new(vec!(hidden_size, hidden_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let wio = Variable::parameter(&Shape::new(vec!(hidden_size, input_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let bo = Variable::parameter(&Shape::new(vec!(hidden_size)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let new_value = tanh(plus(plus(times(&wio, &inputv), &bo), element_times(&reset_gate, times(&woo, &placeholder))));

    let output = plus(element_times(&update_gate, new_value), element_times(minus(one, &update_gate), &placeholder));

    let placeholder_replacement = match init {
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
    let x = Variable::create(&Shape::new(vec!(num_tokens)), true, false, "X", &vec!(&input_axis, &Axis::default_batch_axis()));
    let x2 = Variable::create(&Shape::new(vec!(num_tokens)), true, false, "X2", &vec!(&label_axis, &Axis::default_batch_axis()));
    let y = Variable::create(&Shape::new(vec!(num_tokens)), true, false, "Y", &vec!(&label_axis, &Axis::default_batch_axis()));

    let layer1 = gru_layer(&x, num_tokens, 100, None);
    let layer2 = gru_layer(&layer1, 100, 100, None);

    let layer1_last = last(&layer1);
    let layer2_last = last(&layer2);

    let layer1_predict = gru_layer(&x2, num_tokens, 100, Some(&Variable::from(layer1_last)));
    let layer2_predict = gru_layer(&layer1_predict, 100, 100, Some(&Variable::from(layer2_last)));

    let w2 = Variable::parameter(&Shape::new(vec!(100, num_tokens)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let b = Variable::parameter(&Shape::new(vec!(1, num_tokens)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let noise_w = Variable::constant_repeat(&Shape::new(vec!(num_tokens)), 1.0);

    let loss = reduce_sum(nce_loss(&w2, &b, &layer2_predict, &y, &noise_w, 5), &Axis::all());

    let all_parameters = loss.parameters();
    let learner = Learner::sgd(&all_parameters, &DoubleParameterSchedule::constant(0.01));
    let trainer = Trainer::new(&layer2_predict, &loss, &learner);

    let mut loss_sum = 0.0;
    let batch_size = 10;
    let mut rng = rand::thread_rng();
    let start_range = Range::new(0, translated_tokens.len() - 10);
    let size_range = Range::new(3, 10);
    for iter in 0..1000000 {
        let mut xbatch = Vec::new();
        let mut x2batch = Vec::new();
        let mut ybatch = Vec::new();
        for i in 0..batch_size {
            let pos = start_range.ind_sample(&mut rng);
            let len = size_range.ind_sample(&mut rng);
            let seq = translated_tokens[pos..pos + len].to_owned();
            let mut output = vec!(num_words);
            output.extend(seq.iter().rev());
            let mut expect = vec!();
            expect.extend(seq.iter().rev());
            expect.push(num_words);
            xbatch.push(seq);
            x2batch.push(output);
            ybatch.push(expect);
        }
        let xvalue = Value::batch_of_one_hot_sequences(&Shape::new(vec!(num_tokens)), &xbatch, DeviceDescriptor::cpu());
        let x2value = Value::batch_of_one_hot_sequences(&Shape::new(vec!(num_tokens)), &x2batch, DeviceDescriptor::cpu());
        let yvalue = Value::batch_of_one_hot_sequences(&Shape::new(vec!(num_tokens)), &ybatch, DeviceDescriptor::cpu());

        let datamap = datamap!{&x => &xvalue, &x2 => &x2value, &y => &yvalue};
        let mut outdatamap = outdatamap!{&loss};
        trainer.train_minibatch(&datamap, &mut outdatamap, DeviceDescriptor::cpu());
        let loss_val = outdatamap.get(&loss).unwrap().to_vec();

        loss_sum += loss_val[0];
        if (iter+1) % 100 == 0 {
            println!("iter {} loss val {:?}", iter, loss_sum / 1002.0);
            loss_sum = 0.0;
        }
    }

}