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

fn dist(x: &[f32], y: &[f32]) -> f32 {
    let up: f32 = x.iter().zip(y).map(|(a, b)| a*b).sum();
    let down: f32 = x.iter().map(|a| a*a).sum::<f32>().sqrt();
    let down2: f32 = y.iter().map(|a| a*a).sum::<f32>().sqrt();
    up / down / down2
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

    let x = Variable::sparse_input_variable(&Shape::new(vec!(num_words)));
    let y = Variable::sparse_input_variable(&Shape::new(vec!(num_words)));

    let w1 = Variable::parameter(&Shape::new(vec!(embedding_size, num_words)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());

    let embedded = times(&w1, &x);

    let w2 = Variable::parameter(&Shape::new(vec!(embedding_size, num_words)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let b = Variable::parameter(&Shape::new(vec!(1, num_words)), &ParameterInitializer::glorot_uniform(), DeviceDescriptor::cpu());
    let noise_w = Variable::constant_repeat(&Shape::new(vec!(num_words)), 1.0);

    let loss = reduce_sum(nce_loss(&w2, &b, &embedded, &y, &noise_w, 5), &Axis::all());

    let all_parameters = loss.parameters();
    let learner = Learner::sgd(&all_parameters.iter().collect::<Vec<&Variable>>(), &DoubleParameterSchedule::constant(0.1));
    let trainer = Trainer::new(&embedded, &loss, &learner);
    let mut rng = rand::thread_rng();
    let word_range = Range::new(0, translated_tokens.len());
    let window_range = Range::new(1, 5);

    let prince_id = vocab["prince"];
    println!("prince id {}", prince_id);

    let mut loss_sum = 0.0;
    for iter in 0..1000000 {
        let idata = (0..10).map(|x| word_range.ind_sample(&mut rng)).collect::<Vec<usize>>();
        let odata = idata.iter().map(|x| {
            let mut unclipped = x + window_range.ind_sample(&mut rng);
            if unclipped >= translated_tokens.len() {
                unclipped = translated_tokens.len() - 1;
            }
            translated_tokens[unclipped]
        }).collect::<Vec<usize>>();

        let idata = idata.into_iter().map(|x| translated_tokens[x]).collect::<Vec<usize>>();

        let ivalue = Value::one_hot_seq(&x.shape(), &idata, DeviceDescriptor::cpu());
        let ovalue = Value::one_hot_seq(&y.shape(), &odata, DeviceDescriptor::cpu());
        let datamap = datamap!{&x => &ivalue, &y => &ovalue};
        let mut outdatamap = outdatamap!{&loss};
        trainer.train_minibatch(&datamap, &mut outdatamap, DeviceDescriptor::cpu());
        let loss_val = outdatamap.get(&loss).unwrap().to_vec();

        loss_sum += loss_val[0];
        if (iter+1) % 1000 == 0 {
            println!("loss val {:?}", loss_sum / 1000.0);
            loss_sum = 0.0;
        }
    }

    let embeddings = w1.parameter_to_vec();
    println!("{}", embeddings.len());

    let princeembed = &embeddings[prince_id*embedding_size..(prince_id+1)*embedding_size];
    let mut word_dist = vocab.iter().map(|(word, index)| {
        let embed = &embeddings[index*embedding_size..(index+1)*embedding_size];
        (dist(princeembed, embed), word)
    }).collect::<Vec<(f32, &String)>>();
    word_dist.sort_by(|a, b| a.partial_cmp(b).unwrap());
    //word_dist.sort();
    println!("{:?} {:?}", &word_dist[0..10], &word_dist[word_dist.len()-10..]);
}