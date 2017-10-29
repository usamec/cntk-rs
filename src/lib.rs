#[macro_use]
extern crate cpp;

mod shape;
pub use shape::Shape;

mod variable;
pub use variable::*;

mod function;
pub use function::Function;

mod value;
pub use value::Value;

mod device;
pub use device::DeviceDescriptor;

mod data_map;
pub use data_map::DataMap;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        let var = Variable::input_variable(Shape::from_slice(&vec!(5)));
        let var2 = Variable::input_variable(Shape::from_slice(&vec!(5)));
        let plus = plus(&var, &var2);


        let data: Vec<f32> = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0);
        let data2: Vec<f32> = vec!(11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 110.0);

        let val = Value::batch(&var, &data, DeviceDescriptor::cpu());
        let val2 = Value::batch(&var2, &data2, DeviceDescriptor::cpu());

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
}