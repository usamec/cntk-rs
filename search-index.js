var searchIndex = {};
searchIndex["cntk"] = {"doc":"CNTK","items":[[3,"Shape","cntk","",null,null],[3,"VariableSet","","Wrapper around unordered_set",null,null],[3,"Variable","","",null,null],[3,"ParameterInitializer","","",null,null],[3,"Function","","",null,null],[3,"BackPropState","","",null,null],[3,"Axis","","",null,null],[3,"Value","","",null,null],[3,"DeviceDescriptor","","",null,null],[3,"DataMap","","Wrapper around unordered_map<Variable, Value> to pass bindings to Function evaluation",null,null],[3,"ReplacementMap","","Wrapper around unordered_map<Variable, Variable> to pass replacement for placeholders",null,null],[3,"Learner","","",null,null],[3,"DoubleParameterSchedule","","",null,null],[3,"Trainer","","",null,null],[5,"set_max_num_cpu_threads","","",null,{"inputs":[{"name":"usize"}],"output":null}],[11,"scalar","","",0,{"inputs":[],"output":{"name":"shape"}}],[11,"new","","",0,{"inputs":[{"name":"k"}],"output":{"name":"shape"}}],[11,"total_size","","",0,{"inputs":[{"name":"self"}],"output":{"name":"usize"}}],[11,"rank","","",0,{"inputs":[{"name":"self"}],"output":{"name":"usize"}}],[11,"get","","",0,{"inputs":[{"name":"self"},{"name":"usize"}],"output":{"name":"usize"}}],[11,"append_shape","","",0,{"inputs":[{"name":"self"},{"name":"shape"}],"output":{"name":"shape"}}],[11,"to_vec","","",0,{"inputs":[{"name":"self"}],"output":{"name":"vec"}}],[11,"to_vec_reversed","","",0,{"inputs":[{"name":"self"}],"output":{"name":"vec"}}],[11,"drop","","",0,{"inputs":[{"name":"self"}],"output":null}],[11,"new","","Creates empty VariableSet",1,{"inputs":[],"output":{"name":"variableset"}}],[11,"add","","Adds Variable to set",1,{"inputs":[{"name":"self"},{"name":"t"}],"output":null}],[11,"drop","","",1,{"inputs":[{"name":"self"}],"output":null}],[11,"constant","","",2,{"inputs":[{"name":"f64"}],"output":{"name":"parameterinitializer"}}],[11,"uniform","","",2,{"inputs":[{"name":"f64"}],"output":{"name":"parameterinitializer"}}],[11,"normal","","",2,{"inputs":[{"name":"f64"}],"output":{"name":"parameterinitializer"}}],[11,"xavier","","",2,{"inputs":[],"output":{"name":"parameterinitializer"}}],[11,"glorot_uniform","","",2,{"inputs":[],"output":{"name":"parameterinitializer"}}],[11,"glorot_normal","","",2,{"inputs":[],"output":{"name":"parameterinitializer"}}],[11,"he_uniform","","",2,{"inputs":[],"output":{"name":"parameterinitializer"}}],[11,"he_normal","","",2,{"inputs":[],"output":{"name":"parameterinitializer"}}],[11,"fmt","","",3,{"inputs":[{"name":"self"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"create","","",3,null],[11,"input_variable","","",3,{"inputs":[{"name":"shape"}],"output":{"name":"variable"}}],[11,"sparse_input_variable","","",3,{"inputs":[{"name":"shape"}],"output":{"name":"variable"}}],[11,"input_variable_with_name","","",3,{"inputs":[{"name":"shape"},{"name":"str"}],"output":{"name":"variable"}}],[11,"input_variable_with_gradient","","",3,{"inputs":[{"name":"shape"}],"output":{"name":"variable"}}],[11,"parameter","","",3,{"inputs":[{"name":"shape"},{"name":"parameterinitializer"},{"name":"devicedescriptor"}],"output":{"name":"variable"}}],[11,"placeholder","","",3,{"inputs":[{"name":"shape"}],"output":{"name":"variable"}}],[11,"constant_scalar","","",3,{"inputs":[{"name":"f32"}],"output":{"name":"variable"}}],[11,"constant_repeat","","",3,{"inputs":[{"name":"shape"},{"name":"f32"}],"output":{"name":"variable"}}],[11,"constant_from_slice","","",3,null],[11,"shape","","",3,{"inputs":[{"name":"self"}],"output":{"name":"shape"}}],[11,"is_parameter","","",3,{"inputs":[{"name":"self"}],"output":{"name":"bool"}}],[11,"name","","",3,{"inputs":[{"name":"self"}],"output":{"name":"string"}}],[11,"normal_random","","",3,{"inputs":[{"name":"shape"},{"name":"f64"},{"name":"f64"}],"output":{"name":"variable"}}],[11,"bernoulli_random","","",3,{"inputs":[{"name":"shape"},{"name":"f64"}],"output":{"name":"variable"}}],[11,"uniform_random","","",3,{"inputs":[{"name":"shape"},{"name":"f64"},{"name":"f64"}],"output":{"name":"variable"}}],[11,"gumbel_random","","",3,{"inputs":[{"name":"shape"},{"name":"f64"},{"name":"f64"}],"output":{"name":"variable"}}],[11,"parameter_to_vec","","",3,{"inputs":[{"name":"self"}],"output":{"name":"vec"}}],[11,"clone","","",3,{"inputs":[{"name":"self"}],"output":{"name":"self"}}],[11,"drop","","",3,{"inputs":[{"name":"self"}],"output":null}],[11,"drop","","",2,{"inputs":[{"name":"self"}],"output":null}],[11,"from","","",3,{"inputs":[{"name":"t"}],"output":{"name":"variable"}}],[11,"from","","",3,{"inputs":[{"name":"variable"}],"output":{"name":"variable"}}],[0,"ops","","",null,null],[5,"transpose_axes","cntk::ops","",null,{"inputs":[{"name":"t"},{"name":"axis"},{"name":"axis"}],"output":{"name":"function"}}],[5,"dropout","","",null,{"inputs":[{"name":"t"},{"name":"f64"}],"output":{"name":"function"}}],[5,"splice","","",null,null],[5,"reshape","","",null,{"inputs":[{"name":"t"},{"name":"shape"}],"output":{"name":"function"}}],[5,"slice","","",null,null],[5,"named_alias","","",null,{"inputs":[{"name":"t"},{"name":"str"}],"output":{"name":"function"}}],[5,"past_value","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"future_value","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"past_value_with_init","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[5,"future_value_with_init","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[5,"first","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"last","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"negate","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"sigmoid","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"tanh","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"asin","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"sin","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"acos","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"cos","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"cosh","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"sinh","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"relu","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"exp","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"log","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"square","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"sqrt","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"round","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"floor","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"ceil","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"abs","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"reciprocal","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"softmax","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"hardmax","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"transpose","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"to_batch","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"alias","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"stop_gradient","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"elu","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"leaky_relu","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"softplus","","",null,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[5,"plus","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[5,"minus","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[5,"log_add_exp","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[5,"pow","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[5,"element_times","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[5,"element_divide","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[5,"equal","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[5,"not_equal","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[5,"less","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[5,"less_equal","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[5,"greater","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[5,"greater_equal","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[5,"times","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[5,"transpose_times","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[5,"cosine_distance","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[5,"binary_cross_entropy","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[5,"squared_error","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[5,"cross_entropy_with_softmax","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[5,"classification_error","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[5,"softmax_with_axis","","",null,{"inputs":[{"name":"t"},{"name":"axis"}],"output":{"name":"function"}}],[5,"reduce_sum","","",null,{"inputs":[{"name":"t"},{"name":"axis"}],"output":{"name":"function"}}],[5,"reduce_log_sum","","",null,{"inputs":[{"name":"t"},{"name":"axis"}],"output":{"name":"function"}}],[5,"reduce_mean","","",null,{"inputs":[{"name":"t"},{"name":"axis"}],"output":{"name":"function"}}],[5,"reduce_max","","",null,{"inputs":[{"name":"t"},{"name":"axis"}],"output":{"name":"function"}}],[5,"reduce_min","","",null,{"inputs":[{"name":"t"},{"name":"axis"}],"output":{"name":"function"}}],[5,"reduce_prod","","",null,{"inputs":[{"name":"t"},{"name":"axis"}],"output":{"name":"function"}}],[5,"argmax","","",null,{"inputs":[{"name":"t"},{"name":"axis"}],"output":{"name":"function"}}],[5,"argmin","","",null,{"inputs":[{"name":"t"},{"name":"axis"}],"output":{"name":"function"}}],[5,"normal_random_like","","",null,{"inputs":[{"name":"t"},{"name":"f64"},{"name":"f64"}],"output":{"name":"function"}}],[5,"bernoulli_random_like","","",null,{"inputs":[{"name":"t"},{"name":"f64"}],"output":{"name":"function"}}],[5,"uniform_random_like","","",null,{"inputs":[{"name":"t"},{"name":"f64"},{"name":"f64"}],"output":{"name":"function"}}],[5,"gumbel_random_like","","",null,{"inputs":[{"name":"t"},{"name":"f64"},{"name":"f64"}],"output":{"name":"function"}}],[5,"convolution","","",null,{"inputs":[{"name":"t"},{"name":"u"},{"name":"shape"}],"output":{"name":"function"}}],[5,"max_pooling","","",null,{"inputs":[{"name":"t"},{"name":"shape"},{"name":"shape"}],"output":{"name":"function"}}],[5,"avg_pooling","","",null,{"inputs":[{"name":"t"},{"name":"shape"},{"name":"shape"}],"output":{"name":"function"}}],[5,"clip","","",null,{"inputs":[{"name":"t"},{"name":"u"},{"name":"v"}],"output":{"name":"function"}}],[5,"nce_loss","","",null,{"inputs":[{"name":"t"},{"name":"u"},{"name":"v"},{"name":"w"},{"name":"x"},{"name":"usize"}],"output":{"name":"function"}}],[5,"broadcast_as","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[5,"unpack","","",null,{"inputs":[{"name":"t"},{"name":"f32"}],"output":{"name":"function"}}],[5,"to_sequence_like","","",null,{"inputs":[{"name":"t"},{"name":"u"}],"output":{"name":"function"}}],[11,"fmt","cntk","",4,{"inputs":[{"name":"self"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"from_variable","","",4,{"inputs":[{"name":"t"}],"output":{"name":"function"}}],[11,"combine","","",4,null],[11,"num_outputs","","",4,{"inputs":[{"name":"self"}],"output":{"name":"usize"}}],[11,"to_variable","","",4,{"inputs":[{"name":"self"}],"output":{"name":"result"}}],[11,"evaluate","","",4,{"inputs":[{"name":"self"},{"name":"datamap"},{"name":"datamap"},{"name":"devicedescriptor"}],"output":null}],[11,"forward","","",4,{"inputs":[{"name":"self"},{"name":"datamap"},{"name":"datamap"},{"name":"devicedescriptor"},{"name":"variableset"},{"name":"variableset"}],"output":{"name":"backpropstate"}}],[11,"backward","","",4,{"inputs":[{"name":"self"},{"name":"backpropstate"},{"name":"datamap"},{"name":"datamap"}],"output":null}],[11,"save","","",4,{"inputs":[{"name":"self"},{"name":"str"}],"output":null}],[11,"load","","",4,{"inputs":[{"name":"str"},{"name":"devicedescriptor"}],"output":{"name":"function"}}],[11,"num_inputs","","",4,{"inputs":[{"name":"self"}],"output":{"name":"usize"}}],[11,"inputs","","",4,{"inputs":[{"name":"self"}],"output":{"name":"vec"}}],[11,"outputs","","",4,{"inputs":[{"name":"self"}],"output":{"name":"vec"}}],[11,"replace_placeholders","","",4,{"inputs":[{"name":"self"},{"name":"replacementmap"}],"output":{"name":"function"}}],[11,"num_parameters","","",4,{"inputs":[{"name":"self"}],"output":{"name":"usize"}}],[11,"parameters","","",4,{"inputs":[{"name":"self"}],"output":{"name":"vec"}}],[11,"drop","","",4,{"inputs":[{"name":"self"}],"output":null}],[11,"drop","","",5,{"inputs":[{"name":"self"}],"output":null}],[11,"all","","",6,{"inputs":[],"output":{"name":"axis"}}],[11,"default_batch_axis","","",6,{"inputs":[],"output":{"name":"axis"}}],[11,"named_dynamic","","",6,{"inputs":[{"name":"str"}],"output":{"name":"axis"}}],[11,"all_static","","",6,{"inputs":[],"output":{"name":"axis"}}],[11,"new","","",6,{"inputs":[{"name":"i32"}],"output":{"name":"axis"}}],[11,"drop","","",6,{"inputs":[{"name":"self"}],"output":null}],[11,"clone","","",6,{"inputs":[{"name":"self"}],"output":{"name":"self"}}],[11,"fmt","","",7,{"inputs":[{"name":"self"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"batch_from_vec","","",7,null],[11,"batch_from_ndarray","","",7,{"inputs":[{"name":"shape"},{"name":"arraybase"},{"name":"devicedescriptor"}],"output":{"name":"value"}}],[11,"sequence_from_vec","","",7,null],[11,"sequence_from_ndarray","","",7,{"inputs":[{"name":"shape"},{"name":"arraybase"},{"name":"devicedescriptor"}],"output":{"name":"value"}}],[11,"from_vec","","",7,null],[11,"from_ndarray","","",7,{"inputs":[{"name":"shape"},{"name":"arraybase"},{"name":"devicedescriptor"}],"output":{"name":"value"}}],[11,"one_hot_seq","","",7,null],[11,"batch_of_sequences_from_vec","","",7,null],[11,"batch_of_sequences_from_ndarray","","",7,null],[11,"batch_of_one_hot_sequences","","",7,null],[11,"to_vec","","",7,{"inputs":[{"name":"self"}],"output":{"name":"vec"}}],[11,"to_ndarray","","",7,{"inputs":[{"name":"self"}],"output":{"name":"arrayd"}}],[11,"shape","","",7,{"inputs":[{"name":"self"}],"output":{"name":"shape"}}],[11,"drop","","",7,{"inputs":[{"name":"self"}],"output":null}],[11,"clone","","",8,{"inputs":[{"name":"self"}],"output":{"name":"devicedescriptor"}}],[11,"cpu","","",8,{"inputs":[],"output":{"name":"devicedescriptor"}}],[11,"new","","Creates an empty DataMap",9,{"inputs":[],"output":{"name":"datamap"}}],[11,"add","","Adds binding to DataMap. If mapping for given Variable exists, it will be overwritten.",9,{"inputs":[{"name":"self"},{"name":"t"},{"name":"value"}],"output":null}],[11,"add_null","","Adds binding to null to DataMap. Useful, when we want function evaluation to create the Value.",9,{"inputs":[{"name":"self"},{"name":"t"}],"output":null}],[11,"get","","",9,{"inputs":[{"name":"self"},{"name":"t"}],"output":{"name":"option"}}],[11,"drop","","",9,{"inputs":[{"name":"self"}],"output":null}],[11,"new","","Creates an empty ReplacementMap",10,{"inputs":[],"output":{"name":"replacementmap"}}],[11,"add","","Adds mapping to ReplacementMap. If mapping for given Variable exists, it will be overwritten.",10,{"inputs":[{"name":"self"},{"name":"variable"},{"name":"t"}],"output":null}],[11,"drop","","",10,{"inputs":[{"name":"self"}],"output":null}],[11,"constant","","",11,{"inputs":[{"name":"f64"}],"output":{"name":"doubleparameterschedule"}}],[11,"drop","","",11,{"inputs":[{"name":"self"}],"output":null}],[11,"fmt","","",12,{"inputs":[{"name":"self"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"sgd","","",12,null],[11,"momentum_sgd","","",12,null],[11,"adam","","",12,null],[11,"drop","","",12,{"inputs":[{"name":"self"}],"output":null}],[11,"fmt","","",13,{"inputs":[{"name":"self"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"new","","",13,{"inputs":[{"name":"function"},{"name":"function"},{"name":"learner"}],"output":{"name":"trainer"}}],[11,"new_with_evalatuion","","",13,{"inputs":[{"name":"function"},{"name":"function"},{"name":"function"},{"name":"learner"}],"output":{"name":"trainer"}}],[11,"train_minibatch","","",13,{"inputs":[{"name":"self"},{"name":"datamap"},{"name":"datamap"},{"name":"devicedescriptor"}],"output":null}],[11,"drop","","",13,{"inputs":[{"name":"self"}],"output":null}],[14,"variableset","","",null,null],[14,"datamap","","",null,null],[14,"outdatamap","","",null,null],[14,"replacementmap","","",null,null]],"paths":[[3,"Shape"],[3,"VariableSet"],[3,"ParameterInitializer"],[3,"Variable"],[3,"Function"],[3,"BackPropState"],[3,"Axis"],[3,"Value"],[3,"DeviceDescriptor"],[3,"DataMap"],[3,"ReplacementMap"],[3,"DoubleParameterSchedule"],[3,"Learner"],[3,"Trainer"]]};
initSearch(searchIndex);
