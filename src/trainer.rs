use variable::{Variable};
use learner::Learner;
use function::Function;
use data_map::DataMap;
use device::DeviceDescriptor;
use std::ptr;
use std::ffi::CStr;


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

    pub fn new_with_evalatuion(model: &Function, loss: &Function, evaluation: &Function, learner: &Learner) -> Trainer {
        let modelpayload = model.payload;
        let losspayload = loss.payload;
        let learnerpayload = learner.payload;
        let evaluationpayload = evaluation.payload;
        Trainer { payload: unsafe {
            cpp!([modelpayload as "FunctionPtr", losspayload as "FunctionPtr", evaluationpayload as "FunctionPtr", learnerpayload as "LearnerPtr"] -> TrainerInner as "TrainerPtr"{
                return CreateTrainer(modelpayload, losspayload, evaluationpayload, { learnerpayload });
            })
        }}
    }

    pub fn train_minibatch(&self, arguments: &DataMap, outputs_to_fetch: &mut DataMap, device: DeviceDescriptor) {
        let payload = self.payload;
        let impayload = arguments.payload;
        let mut ompayload = outputs_to_fetch.payload;
        let dpayload = device.payload;
        unsafe {
            let mut error_p: *mut i8 = ptr::null_mut();
            cpp!([payload as "TrainerPtr", impayload as "unordered_map<Variable, ValuePtr>*", mut ompayload as "unordered_map<Variable, ValuePtr>*", dpayload as "DeviceDescriptor", mut error_p as "char*"] {
                try {
                    payload->TrainMinibatch(*impayload, *ompayload, dpayload);
                } catch (std::exception& e) {
                    auto what = e.what();
                    error_p = new char[strlen(what)+1];
                    strcpy(error_p, what);
                }
            });
            if !error_p.is_null() {
                let msg = CStr::from_ptr(error_p).to_str().unwrap();
                panic!("{}", msg);
            }
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