#[macro_use]
extern crate cpp;

//mod bindings;
mod variable;
pub use variable::Variable;

mod function;
pub use function::Function;
pub use function::plus;

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
        loop {
            let var = Variable::new();
            let var2 = Variable::new();
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
            let outputvar = plus.to_variable().unwrap();
            outdatamap.add(&outputvar, &emptyval);

            plus.evaluate(&datamap, &mut outdatamap, DeviceDescriptor::cpu());


        /*    let oval = Value::empty();


            let binding = vec!(
                              Binding(&var, &val),
                              Binding(&var2, &val2)
                              );

            let plusvar = Variable::from(&plus);
            let mut obinding = vec!(
                               Binding(&plusvar, &oval)
                               );
            println!("eval go");
            plus.evaluate(&binding, &mut obinding, DeviceDescriptor::cpu());*/
        }

    }
}