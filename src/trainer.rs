use variable::{Variable};
use learner::Learner;
use function::Function;
use data_map::DataMap;
use device::DeviceDescriptor;

cpp! {{
  #include <CNTKLibrary.h>
  #include <cstdio>
  #include <vector>

  using namespace CNTK;
  using namespace std;
}}

type TrainerInner = [u64; 2usize];

#[derive(Debug)]
pub struct Trainer {
    pub(super) payload: TrainerInner
}

impl Trainer {
    pub fn new(model: &Function, loss: &Function, learner: &Learner) -> Trainer {
        let modelpayload = model.payload;
        let losspayload = loss.payload;
        let learnerpayload = learner.payload;
        Trainer { payload: unsafe {
            cpp!([modelpayload as "FunctionPtr", losspayload as "FunctionPtr", learnerpayload as "LearnerPtr"] -> TrainerInner as "TrainerPtr"{
                return CreateTrainer(modelpayload, losspayload, { learnerpayload });
            })
        }}
    }

    pub fn train_minibatch(&self, arguments: &DataMap, outputs_to_fetch: &mut DataMap, device: DeviceDescriptor) {
        let payload = self.payload;
        let impayload = arguments.payload;
        let mut ompayload = outputs_to_fetch.payload;
        let dpayload = device.payload;
        unsafe {
            cpp!([payload as "TrainerPtr", impayload as "unordered_map<Variable, ValuePtr>*", mut ompayload as "unordered_map<Variable, ValuePtr>*", dpayload as "DeviceDescriptor"] {
                payload->TrainMinibatch(*impayload, *ompayload, dpayload);
            })
        };
    }
}

impl Drop for Trainer {
    fn drop(&mut self) {
        let payload = self.payload;
        unsafe {
            cpp!([payload as "TrainerPtr"] {
                payload.~TrainerPtr();
            })
        };
    }
}