#[macro_use]
extern crate cpp;

mod shape;
pub use shape::Shape;

mod variable_set;
pub use variable_set::VariableSet;

mod variable;
pub use variable::*;

mod function;
pub use function::Function;
pub use function::BackPropState;

mod value;
pub use value::Value;

mod device;
pub use device::{DeviceDescriptor, set_max_num_cpu_threads};

mod data_map;
pub use data_map::DataMap;

mod learner;
pub use learner::Learner;

mod trainer;
pub use trainer::Trainer;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simple_add() {
        let var = Variable::input_variable(Shape::from_slice(&vec!(5)));
        let var2 = Variable::input_variable(Shape::from_slice(&vec!(5)));
        let plus = plus(&var, &var2);

        let data: Vec<f32> = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0);
        let data2: Vec<f32> = vec!(11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 110.0);

        let val = Value::batch(&var.shape(), &data, DeviceDescriptor::cpu());
        let val2 = Value::batch(&var2.shape(), &data2, DeviceDescriptor::cpu());

        let mut datamap = DataMap::new();
        datamap.add(&var, &val);
        datamap.add(&var2, &val2);

        let emptyval = Value::empty();
        let mut outdatamap = DataMap::new();
        outdatamap.add(&plus, &emptyval);

        Function::from_variable(&plus).evaluate(&datamap, &mut outdatamap, DeviceDescriptor::cpu());

        let result = outdatamap.get(&plus).unwrap().to_vec();
        assert_eq!(result, vec!(12., 14., 16., 18., 20., 22., 24., 26., 28., 120.));
    }

    #[test]
    fn gradient() {
        let var = Variable::input_variable_with_gradient(Shape::scalar());
        let var2 = Variable::input_variable_with_gradient(Shape::scalar());
        let var3 = Variable::input_variable_with_gradient(Shape::scalar());
        let out = plus(&elem_times(&var, &var2), &var3);

        let data: Vec<f32> = vec!(4.0, 7.0);
        let data2: Vec<f32> = vec!(11.0, 12.0);
        let data3: Vec<f32> = vec!(11.0, 12.0);

        let val = Value::batch(&var.shape(), &data, DeviceDescriptor::cpu());
        let val2 = Value::batch(&var2.shape(), &data2, DeviceDescriptor::cpu());
        let val3 = Value::batch(&var3.shape(), &data3, DeviceDescriptor::cpu());

        let mut datamap = DataMap::new();
        datamap.add(&var, &val);
        datamap.add(&var2, &val2);
        datamap.add(&var3, &val3);

        let emptyval = Value::empty();
        let mut outdatamap = DataMap::new();
        outdatamap.add(&out, &emptyval);

        let mut retain_state = VariableSet::new();
        retain_state.add(&out);

        let of = Function::from_variable(&out);

        let bpstate = of.forward(&datamap, &mut outdatamap, DeviceDescriptor::cpu(), &retain_state, &VariableSet::new());
        let out_val = outdatamap.get(&out).unwrap();

        let mut result = DataMap::new();
        let emptyval2 = Value::empty();
        let emptyval3 = Value::empty();
        result.add(&var2, &emptyval2);
        result.add(&var3, &emptyval3);

        let mut rgvalues = DataMap::new();
        let rootgrad = Value::from_vec(&out_val.shape(), &(vec![1.; out_val.shape().total_size()]), DeviceDescriptor::cpu());
        rgvalues.add(&out, &rootgrad);

        of.backward(&bpstate, &rgvalues, &mut result);

        assert_eq!(result.get(&var2).unwrap().to_vec(), vec!(4., 7.));
        assert_eq!(result.get(&var3).unwrap().to_vec(), vec!(1., 1.));
    }

    fn rng_next(seed: &mut i32) -> f32 {
        let ret = (*seed % 201 - 100) as f32 / 100.0;
        *seed = seed.wrapping_mul(23);
        ret
    }

    #[test]
    fn feedforward_net_training() {
        set_max_num_cpu_threads(1);
        let x = Variable::input_variable(Shape::from_slice(&vec!(3)));
        let y = Variable::input_variable(Shape::from_slice(&vec!(1)));
        let w1 = Variable::parameter(Shape::from_slice(&vec!(20, 3)), DeviceDescriptor::cpu());
        let b1 = Variable::parameter(Shape::from_slice(&vec!(20)), DeviceDescriptor::cpu());
        let w2 = Variable::parameter(Shape::from_slice(&vec!(1, 20)), DeviceDescriptor::cpu());
        let b2 = Variable::parameter(Shape::from_slice(&vec!(1)), DeviceDescriptor::cpu());

        let hidden_value = tanh(&plus(&times(&w1, &x), &b1));
        let output_value = plus(&times(&w2, &hidden_value), &b2);
        let error = reduce_sum_all(&squared_error(&output_value, &y));

        let predict_func = Function::combine(&vec!(&output_value, &error));
        let output_func = Function::from_variable(&output_value);
        let error_func = Function::from_variable(&error);

        let mut rng_seed = 47;

        let learner = Learner::sgd(&vec!(&w1, &b1, &w2, &b2));
        let trainer = Trainer::new(&output_func, &error_func, &learner);
        let mut lastloss = 1000000.0;

        for iter in 0..50000 {
            let data = vec!(rng_next(&mut rng_seed), rng_next(&mut rng_seed), rng_next(&mut rng_seed),
                            rng_next(&mut rng_seed), rng_next(&mut rng_seed), rng_next(&mut rng_seed));
            let odata = vec!(data[0]*data[1] + data[2],
                             data[3]*data[4] + data[5]);
            let value = Value::batch(&x.shape(), &data, DeviceDescriptor::cpu());
            let ovalue = Value::batch(&y.shape(), &odata, DeviceDescriptor::cpu());
            let mut datamap = DataMap::new();
            datamap.add(&x, &value);
            datamap.add(&y, &ovalue);
            let mut outdatamap = DataMap::new();
            outdatamap.add_null(&output_value);
            outdatamap.add_null(&error);

            trainer.train_minibatch(&datamap, &mut outdatamap, DeviceDescriptor::cpu());
            let result = outdatamap.get(&output_value).unwrap().to_vec();
            let loss = outdatamap.get(&error).unwrap().to_vec();
            lastloss = loss[0];
        }
        assert!(lastloss < 0.1);
    }
}