#[macro_use]
extern crate cntk;
extern crate regex;
extern crate rand;

use cntk::{Variable, Function, Value, Learner, Trainer, DoubleParameterSchedule, DataMap, Axis};
use cntk::ParameterInitializer;
use cntk::Shape;
use cntk::ops::*;
use cntk::DeviceDescriptor;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use regex::Regex;
use std::collections::HashMap;

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

fn main() {
    let file = File::open("data/shakespeare_input.txt").unwrap();
    let mut buf_reader = BufReader::new(file);
    let mut contents = String::new();
    buf_reader.read_to_string(&mut contents).unwrap();
    let tokens = tokenize(&contents);

    let (vocab, translated_tokens) = build_vocab(&tokens);


}