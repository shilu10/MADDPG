§ж
▀«
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58Ѕ¤
б
%Adam/v/critic_network_2/dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/v/critic_network_2/dense_27/bias
Џ
9Adam/v/critic_network_2/dense_27/bias/Read/ReadVariableOpReadVariableOp%Adam/v/critic_network_2/dense_27/bias*
_output_shapes
:*
dtype0
б
%Adam/m/critic_network_2/dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/m/critic_network_2/dense_27/bias
Џ
9Adam/m/critic_network_2/dense_27/bias/Read/ReadVariableOpReadVariableOp%Adam/m/critic_network_2/dense_27/bias*
_output_shapes
:*
dtype0
ф
'Adam/v/critic_network_2/dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *8
shared_name)'Adam/v/critic_network_2/dense_27/kernel
Б
;Adam/v/critic_network_2/dense_27/kernel/Read/ReadVariableOpReadVariableOp'Adam/v/critic_network_2/dense_27/kernel*
_output_shapes

: *
dtype0
ф
'Adam/m/critic_network_2/dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *8
shared_name)'Adam/m/critic_network_2/dense_27/kernel
Б
;Adam/m/critic_network_2/dense_27/kernel/Read/ReadVariableOpReadVariableOp'Adam/m/critic_network_2/dense_27/kernel*
_output_shapes

: *
dtype0
б
%Adam/v/critic_network_2/dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/v/critic_network_2/dense_26/bias
Џ
9Adam/v/critic_network_2/dense_26/bias/Read/ReadVariableOpReadVariableOp%Adam/v/critic_network_2/dense_26/bias*
_output_shapes
: *
dtype0
б
%Adam/m/critic_network_2/dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/m/critic_network_2/dense_26/bias
Џ
9Adam/m/critic_network_2/dense_26/bias/Read/ReadVariableOpReadVariableOp%Adam/m/critic_network_2/dense_26/bias*
_output_shapes
: *
dtype0
ф
'Adam/v/critic_network_2/dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *8
shared_name)'Adam/v/critic_network_2/dense_26/kernel
Б
;Adam/v/critic_network_2/dense_26/kernel/Read/ReadVariableOpReadVariableOp'Adam/v/critic_network_2/dense_26/kernel*
_output_shapes

:  *
dtype0
ф
'Adam/m/critic_network_2/dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *8
shared_name)'Adam/m/critic_network_2/dense_26/kernel
Б
;Adam/m/critic_network_2/dense_26/kernel/Read/ReadVariableOpReadVariableOp'Adam/m/critic_network_2/dense_26/kernel*
_output_shapes

:  *
dtype0
б
%Adam/v/critic_network_2/dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/v/critic_network_2/dense_25/bias
Џ
9Adam/v/critic_network_2/dense_25/bias/Read/ReadVariableOpReadVariableOp%Adam/v/critic_network_2/dense_25/bias*
_output_shapes
: *
dtype0
б
%Adam/m/critic_network_2/dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/m/critic_network_2/dense_25/bias
Џ
9Adam/m/critic_network_2/dense_25/bias/Read/ReadVariableOpReadVariableOp%Adam/m/critic_network_2/dense_25/bias*
_output_shapes
: *
dtype0
ф
'Adam/v/critic_network_2/dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *8
shared_name)'Adam/v/critic_network_2/dense_25/kernel
Б
;Adam/v/critic_network_2/dense_25/kernel/Read/ReadVariableOpReadVariableOp'Adam/v/critic_network_2/dense_25/kernel*
_output_shapes

:@ *
dtype0
ф
'Adam/m/critic_network_2/dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *8
shared_name)'Adam/m/critic_network_2/dense_25/kernel
Б
;Adam/m/critic_network_2/dense_25/kernel/Read/ReadVariableOpReadVariableOp'Adam/m/critic_network_2/dense_25/kernel*
_output_shapes

:@ *
dtype0
б
%Adam/v/critic_network_2/dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%Adam/v/critic_network_2/dense_24/bias
Џ
9Adam/v/critic_network_2/dense_24/bias/Read/ReadVariableOpReadVariableOp%Adam/v/critic_network_2/dense_24/bias*
_output_shapes
:@*
dtype0
б
%Adam/m/critic_network_2/dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%Adam/m/critic_network_2/dense_24/bias
Џ
9Adam/m/critic_network_2/dense_24/bias/Read/ReadVariableOpReadVariableOp%Adam/m/critic_network_2/dense_24/bias*
_output_shapes
:@*
dtype0
ф
'Adam/v/critic_network_2/dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:R@*8
shared_name)'Adam/v/critic_network_2/dense_24/kernel
Б
;Adam/v/critic_network_2/dense_24/kernel/Read/ReadVariableOpReadVariableOp'Adam/v/critic_network_2/dense_24/kernel*
_output_shapes

:R@*
dtype0
ф
'Adam/m/critic_network_2/dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:R@*8
shared_name)'Adam/m/critic_network_2/dense_24/kernel
Б
;Adam/m/critic_network_2/dense_24/kernel/Read/ReadVariableOpReadVariableOp'Adam/m/critic_network_2/dense_24/kernel*
_output_shapes

:R@*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
ћ
critic_network_2/dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name critic_network_2/dense_27/bias
Ї
2critic_network_2/dense_27/bias/Read/ReadVariableOpReadVariableOpcritic_network_2/dense_27/bias*
_output_shapes
:*
dtype0
ю
 critic_network_2/dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *1
shared_name" critic_network_2/dense_27/kernel
Ћ
4critic_network_2/dense_27/kernel/Read/ReadVariableOpReadVariableOp critic_network_2/dense_27/kernel*
_output_shapes

: *
dtype0
ћ
critic_network_2/dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name critic_network_2/dense_26/bias
Ї
2critic_network_2/dense_26/bias/Read/ReadVariableOpReadVariableOpcritic_network_2/dense_26/bias*
_output_shapes
: *
dtype0
ю
 critic_network_2/dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *1
shared_name" critic_network_2/dense_26/kernel
Ћ
4critic_network_2/dense_26/kernel/Read/ReadVariableOpReadVariableOp critic_network_2/dense_26/kernel*
_output_shapes

:  *
dtype0
ћ
critic_network_2/dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name critic_network_2/dense_25/bias
Ї
2critic_network_2/dense_25/bias/Read/ReadVariableOpReadVariableOpcritic_network_2/dense_25/bias*
_output_shapes
: *
dtype0
ю
 critic_network_2/dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *1
shared_name" critic_network_2/dense_25/kernel
Ћ
4critic_network_2/dense_25/kernel/Read/ReadVariableOpReadVariableOp critic_network_2/dense_25/kernel*
_output_shapes

:@ *
dtype0
ћ
critic_network_2/dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name critic_network_2/dense_24/bias
Ї
2critic_network_2/dense_24/bias/Read/ReadVariableOpReadVariableOpcritic_network_2/dense_24/bias*
_output_shapes
:@*
dtype0
ю
 critic_network_2/dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:R@*1
shared_name" critic_network_2/dense_24/kernel
Ћ
4critic_network_2/dense_24/kernel/Read/ReadVariableOpReadVariableOp critic_network_2/dense_24/kernel*
_output_shapes

:R@*
dtype0
y
serving_default_args_0Placeholder*'
_output_shapes
:         >*
dtype0*
shape:         >
y
serving_default_args_1Placeholder*'
_output_shapes
:         *
dtype0*
shape:         
ж
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_1 critic_network_2/dense_24/kernelcritic_network_2/dense_24/bias critic_network_2/dense_25/kernelcritic_network_2/dense_25/bias critic_network_2/dense_26/kernelcritic_network_2/dense_26/bias critic_network_2/dense_27/kernelcritic_network_2/dense_27/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8ѓ */
f*R(
&__inference_signature_wrapper_16873412

NoOpNoOp
│2
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ь1
valueС1Bр1 B┌1
Щ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
fc1
	fc2

fc3
out
	optimizer
loss

signatures*
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*
* 
░
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
д
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

kernel
bias*
д
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

kernel
bias*
д
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

kernel
bias*
д
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

kernel
bias*
Ђ
6
_variables
7_iterations
8_learning_rate
9_index_dict
:
_momentums
;_velocities
<_update_step_xla*
* 

=serving_default* 
`Z
VARIABLE_VALUE critic_network_2/dense_24/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEcritic_network_2/dense_24/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE critic_network_2/dense_25/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEcritic_network_2/dense_25/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE critic_network_2/dense_26/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEcritic_network_2/dense_26/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE critic_network_2/dense_27/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEcritic_network_2/dense_27/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
	1

2
3*
* 
* 
* 
* 
* 

0
1*

0
1*
* 
Њ
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

Ctrace_0* 

Dtrace_0* 

0
1*

0
1*
* 
Њ
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

Jtrace_0* 

Ktrace_0* 

0
1*

0
1*
* 
Њ
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

Qtrace_0* 

Rtrace_0* 

0
1*

0
1*
* 
Њ
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

Xtrace_0* 

Ytrace_0* 
ѓ
70
Z1
[2
\3
]4
^5
_6
`7
a8
b9
c10
d11
e12
f13
g14
h15
i16*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
Z0
\1
^2
`3
b4
d5
f6
h7*
<
[0
]1
_2
a3
c4
e5
g6
i7*
j
jtrace_0
ktrace_1
ltrace_2
mtrace_3
ntrace_4
otrace_5
ptrace_6
qtrace_7* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
rl
VARIABLE_VALUE'Adam/m/critic_network_2/dense_24/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE'Adam/v/critic_network_2/dense_24/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE%Adam/m/critic_network_2/dense_24/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE%Adam/v/critic_network_2/dense_24/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE'Adam/m/critic_network_2/dense_25/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE'Adam/v/critic_network_2/dense_25/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE%Adam/m/critic_network_2/dense_25/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE%Adam/v/critic_network_2/dense_25/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE'Adam/m/critic_network_2/dense_26/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE'Adam/v/critic_network_2/dense_26/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/m/critic_network_2/dense_26/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/v/critic_network_2/dense_26/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE'Adam/m/critic_network_2/dense_27/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE'Adam/v/critic_network_2/dense_27/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/m/critic_network_2/dense_27/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/v/critic_network_2/dense_27/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Т
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename4critic_network_2/dense_24/kernel/Read/ReadVariableOp2critic_network_2/dense_24/bias/Read/ReadVariableOp4critic_network_2/dense_25/kernel/Read/ReadVariableOp2critic_network_2/dense_25/bias/Read/ReadVariableOp4critic_network_2/dense_26/kernel/Read/ReadVariableOp2critic_network_2/dense_26/bias/Read/ReadVariableOp4critic_network_2/dense_27/kernel/Read/ReadVariableOp2critic_network_2/dense_27/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp;Adam/m/critic_network_2/dense_24/kernel/Read/ReadVariableOp;Adam/v/critic_network_2/dense_24/kernel/Read/ReadVariableOp9Adam/m/critic_network_2/dense_24/bias/Read/ReadVariableOp9Adam/v/critic_network_2/dense_24/bias/Read/ReadVariableOp;Adam/m/critic_network_2/dense_25/kernel/Read/ReadVariableOp;Adam/v/critic_network_2/dense_25/kernel/Read/ReadVariableOp9Adam/m/critic_network_2/dense_25/bias/Read/ReadVariableOp9Adam/v/critic_network_2/dense_25/bias/Read/ReadVariableOp;Adam/m/critic_network_2/dense_26/kernel/Read/ReadVariableOp;Adam/v/critic_network_2/dense_26/kernel/Read/ReadVariableOp9Adam/m/critic_network_2/dense_26/bias/Read/ReadVariableOp9Adam/v/critic_network_2/dense_26/bias/Read/ReadVariableOp;Adam/m/critic_network_2/dense_27/kernel/Read/ReadVariableOp;Adam/v/critic_network_2/dense_27/kernel/Read/ReadVariableOp9Adam/m/critic_network_2/dense_27/bias/Read/ReadVariableOp9Adam/v/critic_network_2/dense_27/bias/Read/ReadVariableOpConst*'
Tin 
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ **
f%R#
!__inference__traced_save_16873649
┘	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename critic_network_2/dense_24/kernelcritic_network_2/dense_24/bias critic_network_2/dense_25/kernelcritic_network_2/dense_25/bias critic_network_2/dense_26/kernelcritic_network_2/dense_26/bias critic_network_2/dense_27/kernelcritic_network_2/dense_27/bias	iterationlearning_rate'Adam/m/critic_network_2/dense_24/kernel'Adam/v/critic_network_2/dense_24/kernel%Adam/m/critic_network_2/dense_24/bias%Adam/v/critic_network_2/dense_24/bias'Adam/m/critic_network_2/dense_25/kernel'Adam/v/critic_network_2/dense_25/kernel%Adam/m/critic_network_2/dense_25/bias%Adam/v/critic_network_2/dense_25/bias'Adam/m/critic_network_2/dense_26/kernel'Adam/v/critic_network_2/dense_26/kernel%Adam/m/critic_network_2/dense_26/bias%Adam/v/critic_network_2/dense_26/bias'Adam/m/critic_network_2/dense_27/kernel'Adam/v/critic_network_2/dense_27/kernel%Adam/m/critic_network_2/dense_27/bias%Adam/v/critic_network_2/dense_27/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *-
f(R&
$__inference__traced_restore_16873737Эй
═

═
3__inference_critic_network_2_layer_call_fn_16873434	
state

action
unknown:R@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identityѕбStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallstateactionunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8ѓ *W
fRRP
N__inference_critic_network_2_layer_call_and_return_conditional_losses_16873327o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         >:         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:         >

_user_specified_namestate:OK
'
_output_shapes
:         
 
_user_specified_nameaction
И
O
#__inference__update_step_xla_141129
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
: : *
	_noinline(:H D

_output_shapes

: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
И
O
#__inference__update_step_xla_141119
gradient
variable:  *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:  : *
	_noinline(:H D

_output_shapes

:  
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
╦
ў
+__inference_dense_27_layer_call_fn_16873537

inputs
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dense_27_layer_call_and_return_conditional_losses_16873320o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╔	
э
F__inference_dense_27_layer_call_and_return_conditional_losses_16873320

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
г
K
#__inference__update_step_xla_141124
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
╦
ў
+__inference_dense_25_layer_call_fn_16873497

inputs
unknown:@ 
	unknown_0: 
identityѕбStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_16873287o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
г
K
#__inference__update_step_xla_141114
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ю

э
F__inference_dense_26_layer_call_and_return_conditional_losses_16873304

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
У
ю
N__inference_critic_network_2_layer_call_and_return_conditional_losses_16873327	
state

action#
dense_24_16873271:R@
dense_24_16873273:@#
dense_25_16873288:@ 
dense_25_16873290: #
dense_26_16873305:  
dense_26_16873307: #
dense_27_16873321: 
dense_27_16873323:
identityѕб dense_24/StatefulPartitionedCallб dense_25/StatefulPartitionedCallб dense_26/StatefulPartitionedCallб dense_27/StatefulPartitionedCallM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :r
concatConcatV2stateactionconcat/axis:output:0*
N*
T0*'
_output_shapes
:         Rё
 dense_24/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_24_16873271dense_24_16873273*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dense_24_layer_call_and_return_conditional_losses_16873270ъ
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_16873288dense_25_16873290*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_16873287ъ
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0dense_26_16873305dense_26_16873307*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dense_26_layer_call_and_return_conditional_losses_16873304ъ
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_16873321dense_27_16873323*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dense_27_layer_call_and_return_conditional_losses_16873320x
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         м
NoOpNoOp!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         >:         : : : : : : : : 2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall:N J
'
_output_shapes
:         >

_user_specified_namestate:OK
'
_output_shapes
:         
 
_user_specified_nameaction
ъ&
л
N__inference_critic_network_2_layer_call_and_return_conditional_losses_16873468	
state

action9
'dense_24_matmul_readvariableop_resource:R@6
(dense_24_biasadd_readvariableop_resource:@9
'dense_25_matmul_readvariableop_resource:@ 6
(dense_25_biasadd_readvariableop_resource: 9
'dense_26_matmul_readvariableop_resource:  6
(dense_26_biasadd_readvariableop_resource: 9
'dense_27_matmul_readvariableop_resource: 6
(dense_27_biasadd_readvariableop_resource:
identityѕбdense_24/BiasAdd/ReadVariableOpбdense_24/MatMul/ReadVariableOpбdense_25/BiasAdd/ReadVariableOpбdense_25/MatMul/ReadVariableOpбdense_26/BiasAdd/ReadVariableOpбdense_26/MatMul/ReadVariableOpбdense_27/BiasAdd/ReadVariableOpбdense_27/MatMul/ReadVariableOpM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :r
concatConcatV2stateactionconcat/axis:output:0*
N*
T0*'
_output_shapes
:         Rє
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:R@*
dtype0ё
dense_24/MatMulMatMulconcat:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ё
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Љ
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @b
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*'
_output_shapes
:         @є
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0љ
dense_25/MatMulMatMuldense_24/Relu:activations:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ё
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Љ
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          b
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*'
_output_shapes
:          є
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0љ
dense_26/MatMulMatMuldense_25/Relu:activations:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ё
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Љ
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          b
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:          є
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

: *
dtype0љ
dense_27/MatMulMatMuldense_26/Relu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_27/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         м
NoOpNoOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         >:         : : : : : : : : 2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp:N J
'
_output_shapes
:         >

_user_specified_namestate:OK
'
_output_shapes
:         
 
_user_specified_nameaction
г
K
#__inference__update_step_xla_141104
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
г
K
#__inference__update_step_xla_141134
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ю

э
F__inference_dense_26_layer_call_and_return_conditional_losses_16873528

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
ю?
Х
!__inference__traced_save_16873649
file_prefix?
;savev2_critic_network_2_dense_24_kernel_read_readvariableop=
9savev2_critic_network_2_dense_24_bias_read_readvariableop?
;savev2_critic_network_2_dense_25_kernel_read_readvariableop=
9savev2_critic_network_2_dense_25_bias_read_readvariableop?
;savev2_critic_network_2_dense_26_kernel_read_readvariableop=
9savev2_critic_network_2_dense_26_bias_read_readvariableop?
;savev2_critic_network_2_dense_27_kernel_read_readvariableop=
9savev2_critic_network_2_dense_27_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableopF
Bsavev2_adam_m_critic_network_2_dense_24_kernel_read_readvariableopF
Bsavev2_adam_v_critic_network_2_dense_24_kernel_read_readvariableopD
@savev2_adam_m_critic_network_2_dense_24_bias_read_readvariableopD
@savev2_adam_v_critic_network_2_dense_24_bias_read_readvariableopF
Bsavev2_adam_m_critic_network_2_dense_25_kernel_read_readvariableopF
Bsavev2_adam_v_critic_network_2_dense_25_kernel_read_readvariableopD
@savev2_adam_m_critic_network_2_dense_25_bias_read_readvariableopD
@savev2_adam_v_critic_network_2_dense_25_bias_read_readvariableopF
Bsavev2_adam_m_critic_network_2_dense_26_kernel_read_readvariableopF
Bsavev2_adam_v_critic_network_2_dense_26_kernel_read_readvariableopD
@savev2_adam_m_critic_network_2_dense_26_bias_read_readvariableopD
@savev2_adam_v_critic_network_2_dense_26_bias_read_readvariableopF
Bsavev2_adam_m_critic_network_2_dense_27_kernel_read_readvariableopF
Bsavev2_adam_v_critic_network_2_dense_27_kernel_read_readvariableopD
@savev2_adam_m_critic_network_2_dense_27_bias_read_readvariableopD
@savev2_adam_v_critic_network_2_dense_27_bias_read_readvariableop
savev2_const

identity_1ѕбMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partЂ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: У

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Љ

valueЄ
Bё
B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHБ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B м
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0;savev2_critic_network_2_dense_24_kernel_read_readvariableop9savev2_critic_network_2_dense_24_bias_read_readvariableop;savev2_critic_network_2_dense_25_kernel_read_readvariableop9savev2_critic_network_2_dense_25_bias_read_readvariableop;savev2_critic_network_2_dense_26_kernel_read_readvariableop9savev2_critic_network_2_dense_26_bias_read_readvariableop;savev2_critic_network_2_dense_27_kernel_read_readvariableop9savev2_critic_network_2_dense_27_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableopBsavev2_adam_m_critic_network_2_dense_24_kernel_read_readvariableopBsavev2_adam_v_critic_network_2_dense_24_kernel_read_readvariableop@savev2_adam_m_critic_network_2_dense_24_bias_read_readvariableop@savev2_adam_v_critic_network_2_dense_24_bias_read_readvariableopBsavev2_adam_m_critic_network_2_dense_25_kernel_read_readvariableopBsavev2_adam_v_critic_network_2_dense_25_kernel_read_readvariableop@savev2_adam_m_critic_network_2_dense_25_bias_read_readvariableop@savev2_adam_v_critic_network_2_dense_25_bias_read_readvariableopBsavev2_adam_m_critic_network_2_dense_26_kernel_read_readvariableopBsavev2_adam_v_critic_network_2_dense_26_kernel_read_readvariableop@savev2_adam_m_critic_network_2_dense_26_bias_read_readvariableop@savev2_adam_v_critic_network_2_dense_26_bias_read_readvariableopBsavev2_adam_m_critic_network_2_dense_27_kernel_read_readvariableopBsavev2_adam_v_critic_network_2_dense_27_kernel_read_readvariableop@savev2_adam_m_critic_network_2_dense_27_bias_read_readvariableop@savev2_adam_v_critic_network_2_dense_27_bias_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *)
dtypes
2	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*П
_input_shapes╦
╚: :R@:@:@ : :  : : :: : :R@:R@:@:@:@ :@ : : :  :  : : : : ::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:R@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :$ 

_output_shapes

:R@:$ 

_output_shapes

:R@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@ :$ 

_output_shapes

:@ : 

_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

:  :$ 

_output_shapes

:  : 

_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

: :$ 

_output_shapes

: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
Ю

э
F__inference_dense_24_layer_call_and_return_conditional_losses_16873488

inputs0
matmul_readvariableop_resource:R@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:R@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         R: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         R
 
_user_specified_nameinputs
ў

┴
&__inference_signature_wrapper_16873412

args_0

args_1
unknown:R@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identityѕбStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallargs_0args_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8ѓ *,
f'R%
#__inference__wrapped_model_16873248o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         >:         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         >
 
_user_specified_nameargs_0:OK
'
_output_shapes
:         
 
_user_specified_nameargs_1
╦
ў
+__inference_dense_24_layer_call_fn_16873477

inputs
unknown:R@
	unknown_0:@
identityѕбStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dense_24_layer_call_and_return_conditional_losses_16873270o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         R: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         R
 
_user_specified_nameinputs
Ю

э
F__inference_dense_24_layer_call_and_return_conditional_losses_16873270

inputs0
matmul_readvariableop_resource:R@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:R@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         R: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         R
 
_user_specified_nameinputs
З1
Х
#__inference__wrapped_model_16873248

args_0

args_1J
8critic_network_2_dense_24_matmul_readvariableop_resource:R@G
9critic_network_2_dense_24_biasadd_readvariableop_resource:@J
8critic_network_2_dense_25_matmul_readvariableop_resource:@ G
9critic_network_2_dense_25_biasadd_readvariableop_resource: J
8critic_network_2_dense_26_matmul_readvariableop_resource:  G
9critic_network_2_dense_26_biasadd_readvariableop_resource: J
8critic_network_2_dense_27_matmul_readvariableop_resource: G
9critic_network_2_dense_27_biasadd_readvariableop_resource:
identityѕб0critic_network_2/dense_24/BiasAdd/ReadVariableOpб/critic_network_2/dense_24/MatMul/ReadVariableOpб0critic_network_2/dense_25/BiasAdd/ReadVariableOpб/critic_network_2/dense_25/MatMul/ReadVariableOpб0critic_network_2/dense_26/BiasAdd/ReadVariableOpб/critic_network_2/dense_26/MatMul/ReadVariableOpб0critic_network_2/dense_27/BiasAdd/ReadVariableOpб/critic_network_2/dense_27/MatMul/ReadVariableOp^
critic_network_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ћ
critic_network_2/concatConcatV2args_0args_1%critic_network_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:         Rе
/critic_network_2/dense_24/MatMul/ReadVariableOpReadVariableOp8critic_network_2_dense_24_matmul_readvariableop_resource*
_output_shapes

:R@*
dtype0и
 critic_network_2/dense_24/MatMulMatMul critic_network_2/concat:output:07critic_network_2/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @д
0critic_network_2/dense_24/BiasAdd/ReadVariableOpReadVariableOp9critic_network_2_dense_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0─
!critic_network_2/dense_24/BiasAddBiasAdd*critic_network_2/dense_24/MatMul:product:08critic_network_2/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ё
critic_network_2/dense_24/ReluRelu*critic_network_2/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:         @е
/critic_network_2/dense_25/MatMul/ReadVariableOpReadVariableOp8critic_network_2_dense_25_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0├
 critic_network_2/dense_25/MatMulMatMul,critic_network_2/dense_24/Relu:activations:07critic_network_2/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          д
0critic_network_2/dense_25/BiasAdd/ReadVariableOpReadVariableOp9critic_network_2_dense_25_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0─
!critic_network_2/dense_25/BiasAddBiasAdd*critic_network_2/dense_25/MatMul:product:08critic_network_2/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ё
critic_network_2/dense_25/ReluRelu*critic_network_2/dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:          е
/critic_network_2/dense_26/MatMul/ReadVariableOpReadVariableOp8critic_network_2_dense_26_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0├
 critic_network_2/dense_26/MatMulMatMul,critic_network_2/dense_25/Relu:activations:07critic_network_2/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          д
0critic_network_2/dense_26/BiasAdd/ReadVariableOpReadVariableOp9critic_network_2_dense_26_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0─
!critic_network_2/dense_26/BiasAddBiasAdd*critic_network_2/dense_26/MatMul:product:08critic_network_2/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ё
critic_network_2/dense_26/ReluRelu*critic_network_2/dense_26/BiasAdd:output:0*
T0*'
_output_shapes
:          е
/critic_network_2/dense_27/MatMul/ReadVariableOpReadVariableOp8critic_network_2_dense_27_matmul_readvariableop_resource*
_output_shapes

: *
dtype0├
 critic_network_2/dense_27/MatMulMatMul,critic_network_2/dense_26/Relu:activations:07critic_network_2/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         д
0critic_network_2/dense_27/BiasAdd/ReadVariableOpReadVariableOp9critic_network_2_dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0─
!critic_network_2/dense_27/BiasAddBiasAdd*critic_network_2/dense_27/MatMul:product:08critic_network_2/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         y
IdentityIdentity*critic_network_2/dense_27/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ┌
NoOpNoOp1^critic_network_2/dense_24/BiasAdd/ReadVariableOp0^critic_network_2/dense_24/MatMul/ReadVariableOp1^critic_network_2/dense_25/BiasAdd/ReadVariableOp0^critic_network_2/dense_25/MatMul/ReadVariableOp1^critic_network_2/dense_26/BiasAdd/ReadVariableOp0^critic_network_2/dense_26/MatMul/ReadVariableOp1^critic_network_2/dense_27/BiasAdd/ReadVariableOp0^critic_network_2/dense_27/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         >:         : : : : : : : : 2d
0critic_network_2/dense_24/BiasAdd/ReadVariableOp0critic_network_2/dense_24/BiasAdd/ReadVariableOp2b
/critic_network_2/dense_24/MatMul/ReadVariableOp/critic_network_2/dense_24/MatMul/ReadVariableOp2d
0critic_network_2/dense_25/BiasAdd/ReadVariableOp0critic_network_2/dense_25/BiasAdd/ReadVariableOp2b
/critic_network_2/dense_25/MatMul/ReadVariableOp/critic_network_2/dense_25/MatMul/ReadVariableOp2d
0critic_network_2/dense_26/BiasAdd/ReadVariableOp0critic_network_2/dense_26/BiasAdd/ReadVariableOp2b
/critic_network_2/dense_26/MatMul/ReadVariableOp/critic_network_2/dense_26/MatMul/ReadVariableOp2d
0critic_network_2/dense_27/BiasAdd/ReadVariableOp0critic_network_2/dense_27/BiasAdd/ReadVariableOp2b
/critic_network_2/dense_27/MatMul/ReadVariableOp/critic_network_2/dense_27/MatMul/ReadVariableOp:O K
'
_output_shapes
:         >
 
_user_specified_nameargs_0:OK
'
_output_shapes
:         
 
_user_specified_nameargs_1
╦
ў
+__inference_dense_26_layer_call_fn_16873517

inputs
unknown:  
	unknown_0: 
identityѕбStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dense_26_layer_call_and_return_conditional_losses_16873304o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
И
O
#__inference__update_step_xla_141099
gradient
variable:R@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:R@: *
	_noinline(:H D

_output_shapes

:R@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
╔	
э
F__inference_dense_27_layer_call_and_return_conditional_losses_16873547

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Ю

э
F__inference_dense_25_layer_call_and_return_conditional_losses_16873287

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
И
O
#__inference__update_step_xla_141109
gradient
variable:@ *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:@ : *
	_noinline(:H D

_output_shapes

:@ 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
░u
Г
$__inference__traced_restore_16873737
file_prefixC
1assignvariableop_critic_network_2_dense_24_kernel:R@?
1assignvariableop_1_critic_network_2_dense_24_bias:@E
3assignvariableop_2_critic_network_2_dense_25_kernel:@ ?
1assignvariableop_3_critic_network_2_dense_25_bias: E
3assignvariableop_4_critic_network_2_dense_26_kernel:  ?
1assignvariableop_5_critic_network_2_dense_26_bias: E
3assignvariableop_6_critic_network_2_dense_27_kernel: ?
1assignvariableop_7_critic_network_2_dense_27_bias:&
assignvariableop_8_iteration:	 *
 assignvariableop_9_learning_rate: M
;assignvariableop_10_adam_m_critic_network_2_dense_24_kernel:R@M
;assignvariableop_11_adam_v_critic_network_2_dense_24_kernel:R@G
9assignvariableop_12_adam_m_critic_network_2_dense_24_bias:@G
9assignvariableop_13_adam_v_critic_network_2_dense_24_bias:@M
;assignvariableop_14_adam_m_critic_network_2_dense_25_kernel:@ M
;assignvariableop_15_adam_v_critic_network_2_dense_25_kernel:@ G
9assignvariableop_16_adam_m_critic_network_2_dense_25_bias: G
9assignvariableop_17_adam_v_critic_network_2_dense_25_bias: M
;assignvariableop_18_adam_m_critic_network_2_dense_26_kernel:  M
;assignvariableop_19_adam_v_critic_network_2_dense_26_kernel:  G
9assignvariableop_20_adam_m_critic_network_2_dense_26_bias: G
9assignvariableop_21_adam_v_critic_network_2_dense_26_bias: M
;assignvariableop_22_adam_m_critic_network_2_dense_27_kernel: M
;assignvariableop_23_adam_v_critic_network_2_dense_27_kernel: G
9assignvariableop_24_adam_m_critic_network_2_dense_27_bias:G
9assignvariableop_25_adam_v_critic_network_2_dense_27_bias:
identity_27ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9в

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Љ

valueЄ
Bё
B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHд
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B д
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ђ
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOpAssignVariableOp1assignvariableop_critic_network_2_dense_24_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_1AssignVariableOp1assignvariableop_1_critic_network_2_dense_24_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_2AssignVariableOp3assignvariableop_2_critic_network_2_dense_25_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_3AssignVariableOp1assignvariableop_3_critic_network_2_dense_25_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_4AssignVariableOp3assignvariableop_4_critic_network_2_dense_26_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_5AssignVariableOp1assignvariableop_5_critic_network_2_dense_26_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_6AssignVariableOp3assignvariableop_6_critic_network_2_dense_27_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_7AssignVariableOp1assignvariableop_7_critic_network_2_dense_27_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:│
AssignVariableOp_8AssignVariableOpassignvariableop_8_iterationIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_9AssignVariableOp assignvariableop_9_learning_rateIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOp_10AssignVariableOp;assignvariableop_10_adam_m_critic_network_2_dense_24_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOp_11AssignVariableOp;assignvariableop_11_adam_v_critic_network_2_dense_24_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:м
AssignVariableOp_12AssignVariableOp9assignvariableop_12_adam_m_critic_network_2_dense_24_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:м
AssignVariableOp_13AssignVariableOp9assignvariableop_13_adam_v_critic_network_2_dense_24_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOp_14AssignVariableOp;assignvariableop_14_adam_m_critic_network_2_dense_25_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOp_15AssignVariableOp;assignvariableop_15_adam_v_critic_network_2_dense_25_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:м
AssignVariableOp_16AssignVariableOp9assignvariableop_16_adam_m_critic_network_2_dense_25_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:м
AssignVariableOp_17AssignVariableOp9assignvariableop_17_adam_v_critic_network_2_dense_25_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOp_18AssignVariableOp;assignvariableop_18_adam_m_critic_network_2_dense_26_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOp_19AssignVariableOp;assignvariableop_19_adam_v_critic_network_2_dense_26_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:м
AssignVariableOp_20AssignVariableOp9assignvariableop_20_adam_m_critic_network_2_dense_26_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:м
AssignVariableOp_21AssignVariableOp9assignvariableop_21_adam_v_critic_network_2_dense_26_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOp_22AssignVariableOp;assignvariableop_22_adam_m_critic_network_2_dense_27_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOp_23AssignVariableOp;assignvariableop_23_adam_v_critic_network_2_dense_27_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:м
AssignVariableOp_24AssignVariableOp9assignvariableop_24_adam_m_critic_network_2_dense_27_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:м
AssignVariableOp_25AssignVariableOp9assignvariableop_25_adam_v_critic_network_2_dense_27_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 І
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_27IdentityIdentity_26:output:0^NoOp_1*
T0*
_output_shapes
: Э
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_27Identity_27:output:0*I
_input_shapes8
6: : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ю

э
F__inference_dense_25_layer_call_and_return_conditional_losses_16873508

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs"є
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*С
serving_defaultл
9
args_0/
serving_default_args_0:0         >
9
args_1/
serving_default_args_1:0         <
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:Зѕ
Ј
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
fc1
	fc2

fc3
out
	optimizer
loss

signatures"
_tf_keras_model
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ђ
trace_02с
3__inference_critic_network_2_layer_call_fn_16873434Ф
б▓ъ
FullArgSpec&
argsџ
jself
jstate
jaction
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 ztrace_0
Џ
trace_02■
N__inference_critic_network_2_layer_call_and_return_conditional_losses_16873468Ф
б▓ъ
FullArgSpec&
argsџ
jself
jstate
jaction
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 ztrace_0
НBм
#__inference__wrapped_model_16873248args_0args_1"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╗
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
╗
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
╗
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
╗
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ю
6
_variables
7_iterations
8_learning_rate
9_index_dict
:
_momentums
;_velocities
<_update_step_xla"
experimentalOptimizer
 "
trackable_dict_wrapper
,
=serving_default"
signature_map
2:0R@2 critic_network_2/dense_24/kernel
,:*@2critic_network_2/dense_24/bias
2:0@ 2 critic_network_2/dense_25/kernel
,:* 2critic_network_2/dense_25/bias
2:0  2 critic_network_2/dense_26/kernel
,:* 2critic_network_2/dense_26/bias
2:0 2 critic_network_2/dense_27/kernel
,:*2critic_network_2/dense_27/bias
 "
trackable_list_wrapper
<
0
	1

2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
эBЗ
3__inference_critic_network_2_layer_call_fn_16873434stateaction"Ф
б▓ъ
FullArgSpec&
argsџ
jself
jstate
jaction
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
њBЈ
N__inference_critic_network_2_layer_call_and_return_conditional_losses_16873468stateaction"Ф
б▓ъ
FullArgSpec&
argsџ
jself
jstate
jaction
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
№
Ctrace_02м
+__inference_dense_24_layer_call_fn_16873477б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zCtrace_0
і
Dtrace_02ь
F__inference_dense_24_layer_call_and_return_conditional_losses_16873488б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zDtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
№
Jtrace_02м
+__inference_dense_25_layer_call_fn_16873497б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zJtrace_0
і
Ktrace_02ь
F__inference_dense_25_layer_call_and_return_conditional_losses_16873508б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zKtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
№
Qtrace_02м
+__inference_dense_26_layer_call_fn_16873517б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zQtrace_0
і
Rtrace_02ь
F__inference_dense_26_layer_call_and_return_conditional_losses_16873528б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zRtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
№
Xtrace_02м
+__inference_dense_27_layer_call_fn_16873537б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zXtrace_0
і
Ytrace_02ь
F__inference_dense_27_layer_call_and_return_conditional_losses_16873547б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zYtrace_0
ъ
70
Z1
[2
\3
]4
^5
_6
`7
a8
b9
c10
d11
e12
f13
g14
h15
i16"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
X
Z0
\1
^2
`3
b4
d5
f6
h7"
trackable_list_wrapper
X
[0
]1
_2
a3
c4
e5
g6
i7"
trackable_list_wrapper
и
jtrace_0
ktrace_1
ltrace_2
mtrace_3
ntrace_4
otrace_5
ptrace_6
qtrace_72С
#__inference__update_step_xla_141099
#__inference__update_step_xla_141104
#__inference__update_step_xla_141109
#__inference__update_step_xla_141114
#__inference__update_step_xla_141119
#__inference__update_step_xla_141124
#__inference__update_step_xla_141129
#__inference__update_step_xla_141134╣
«▓ф
FullArgSpec2
args*џ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0zjtrace_0zktrace_1zltrace_2zmtrace_3zntrace_4zotrace_5zptrace_6zqtrace_7
мB¤
&__inference_signature_wrapper_16873412args_0args_1"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_dense_24_layer_call_fn_16873477inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
F__inference_dense_24_layer_call_and_return_conditional_losses_16873488inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_dense_25_layer_call_fn_16873497inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
F__inference_dense_25_layer_call_and_return_conditional_losses_16873508inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_dense_26_layer_call_fn_16873517inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
F__inference_dense_26_layer_call_and_return_conditional_losses_16873528inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_dense_27_layer_call_fn_16873537inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
F__inference_dense_27_layer_call_and_return_conditional_losses_16873547inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
7:5R@2'Adam/m/critic_network_2/dense_24/kernel
7:5R@2'Adam/v/critic_network_2/dense_24/kernel
1:/@2%Adam/m/critic_network_2/dense_24/bias
1:/@2%Adam/v/critic_network_2/dense_24/bias
7:5@ 2'Adam/m/critic_network_2/dense_25/kernel
7:5@ 2'Adam/v/critic_network_2/dense_25/kernel
1:/ 2%Adam/m/critic_network_2/dense_25/bias
1:/ 2%Adam/v/critic_network_2/dense_25/bias
7:5  2'Adam/m/critic_network_2/dense_26/kernel
7:5  2'Adam/v/critic_network_2/dense_26/kernel
1:/ 2%Adam/m/critic_network_2/dense_26/bias
1:/ 2%Adam/v/critic_network_2/dense_26/bias
7:5 2'Adam/m/critic_network_2/dense_27/kernel
7:5 2'Adam/v/critic_network_2/dense_27/kernel
1:/2%Adam/m/critic_network_2/dense_27/bias
1:/2%Adam/v/critic_network_2/dense_27/bias
ЭBш
#__inference__update_step_xla_141099gradientvariable"и
«▓ф
FullArgSpec2
args*џ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
#__inference__update_step_xla_141104gradientvariable"и
«▓ф
FullArgSpec2
args*џ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
#__inference__update_step_xla_141109gradientvariable"и
«▓ф
FullArgSpec2
args*џ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
#__inference__update_step_xla_141114gradientvariable"и
«▓ф
FullArgSpec2
args*џ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
#__inference__update_step_xla_141119gradientvariable"и
«▓ф
FullArgSpec2
args*џ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
#__inference__update_step_xla_141124gradientvariable"и
«▓ф
FullArgSpec2
args*џ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
#__inference__update_step_xla_141129gradientvariable"и
«▓ф
FullArgSpec2
args*џ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
#__inference__update_step_xla_141134gradientvariable"и
«▓ф
FullArgSpec2
args*џ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 Ћ
#__inference__update_step_xla_141099nhбe
^б[
і
gradientR@
4њ1	б
ЩR@
ђ
p
` VariableSpec 
`ахѕўя▒?
ф "
 Ї
#__inference__update_step_xla_141104f`б]
VбS
і
gradient@
0њ-	б
Щ@
ђ
p
` VariableSpec 
`Я│ѕўя▒?
ф "
 Ћ
#__inference__update_step_xla_141109nhбe
^б[
і
gradient@ 
4њ1	б
Щ@ 
ђ
p
` VariableSpec 
`ЯЃ╠Ќя▒?
ф "
 Ї
#__inference__update_step_xla_141114f`б]
VбS
і
gradient 
0њ-	б
Щ 
ђ
p
` VariableSpec 
`аѓ╠Ќя▒?
ф "
 Ћ
#__inference__update_step_xla_141119nhбe
^б[
і
gradient  
4њ1	б
Щ  
ђ
p
` VariableSpec 
`аЉ╠Ќя▒?
ф "
 Ї
#__inference__update_step_xla_141124f`б]
VбS
і
gradient 
0њ-	б
Щ 
ђ
p
` VariableSpec 
`ЯЈ╠Ќя▒?
ф "
 Ћ
#__inference__update_step_xla_141129nhбe
^б[
і
gradient 
4њ1	б
Щ 
ђ
p
` VariableSpec 
`Яъ╠Ќя▒?
ф "
 Ї
#__inference__update_step_xla_141134f`б]
VбS
і
gradient
0њ-	б
Щ
ђ
p
` VariableSpec 
`аЮ╠Ќя▒?
ф "
 ║
#__inference__wrapped_model_16873248њQбN
GбD
 і
args_0         >
 і
args_1         
ф "3ф0
.
output_1"і
output_1         П
N__inference_critic_network_2_layer_call_and_return_conditional_losses_16873468іPбM
FбC
і
state         >
 і
action         
ф ",б)
"і
tensor_0         
џ Х
3__inference_critic_network_2_layer_call_fn_16873434PбM
FбC
і
state         >
 і
action         
ф "!і
unknown         Г
F__inference_dense_24_layer_call_and_return_conditional_losses_16873488c/б,
%б"
 і
inputs         R
ф ",б)
"і
tensor_0         @
џ Є
+__inference_dense_24_layer_call_fn_16873477X/б,
%б"
 і
inputs         R
ф "!і
unknown         @Г
F__inference_dense_25_layer_call_and_return_conditional_losses_16873508c/б,
%б"
 і
inputs         @
ф ",б)
"і
tensor_0          
џ Є
+__inference_dense_25_layer_call_fn_16873497X/б,
%б"
 і
inputs         @
ф "!і
unknown          Г
F__inference_dense_26_layer_call_and_return_conditional_losses_16873528c/б,
%б"
 і
inputs          
ф ",б)
"і
tensor_0          
џ Є
+__inference_dense_26_layer_call_fn_16873517X/б,
%б"
 і
inputs          
ф "!і
unknown          Г
F__inference_dense_27_layer_call_and_return_conditional_losses_16873547c/б,
%б"
 і
inputs          
ф ",б)
"і
tensor_0         
џ Є
+__inference_dense_27_layer_call_fn_16873537X/б,
%б"
 і
inputs          
ф "!і
unknown         Л
&__inference_signature_wrapper_16873412дeбb
б 
[фX
*
args_0 і
args_0         >
*
args_1 і
args_1         "3ф0
.
output_1"і
output_1         