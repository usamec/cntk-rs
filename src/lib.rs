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
pub use device::DeviceDescriptor;

mod data_map;
pub use data_map::DataMap;

#[cfg(test)]
mod tests {
    use super::*;
    #[test] #[ignore]
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
}