�
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring ��
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58Ő
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
�
critic_network_7/dense_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name critic_network_7/dense_63/bias
�
2critic_network_7/dense_63/bias/Read/ReadVariableOpReadVariableOpcritic_network_7/dense_63/bias*
_output_shapes
:*
dtype0
�
 critic_network_7/dense_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *1
shared_name" critic_network_7/dense_63/kernel
�
4critic_network_7/dense_63/kernel/Read/ReadVariableOpReadVariableOp critic_network_7/dense_63/kernel*
_output_shapes

: *
dtype0
�
critic_network_7/dense_62/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name critic_network_7/dense_62/bias
�
2critic_network_7/dense_62/bias/Read/ReadVariableOpReadVariableOpcritic_network_7/dense_62/bias*
_output_shapes
: *
dtype0
�
 critic_network_7/dense_62/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *1
shared_name" critic_network_7/dense_62/kernel
�
4critic_network_7/dense_62/kernel/Read/ReadVariableOpReadVariableOp critic_network_7/dense_62/kernel*
_output_shapes

:  *
dtype0
�
critic_network_7/dense_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name critic_network_7/dense_61/bias
�
2critic_network_7/dense_61/bias/Read/ReadVariableOpReadVariableOpcritic_network_7/dense_61/bias*
_output_shapes
: *
dtype0
�
 critic_network_7/dense_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *1
shared_name" critic_network_7/dense_61/kernel
�
4critic_network_7/dense_61/kernel/Read/ReadVariableOpReadVariableOp critic_network_7/dense_61/kernel*
_output_shapes

:@ *
dtype0
�
critic_network_7/dense_60/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name critic_network_7/dense_60/bias
�
2critic_network_7/dense_60/bias/Read/ReadVariableOpReadVariableOpcritic_network_7/dense_60/bias*
_output_shapes
:@*
dtype0
�
 critic_network_7/dense_60/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:R@*1
shared_name" critic_network_7/dense_60/kernel
�
4critic_network_7/dense_60/kernel/Read/ReadVariableOpReadVariableOp critic_network_7/dense_60/kernel*
_output_shapes

:R@*
dtype0
y
serving_default_args_0Placeholder*'
_output_shapes
:���������>*
dtype0*
shape:���������>
y
serving_default_args_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_1 critic_network_7/dense_60/kernelcritic_network_7/dense_60/bias critic_network_7/dense_61/kernelcritic_network_7/dense_61/bias critic_network_7/dense_62/kernelcritic_network_7/dense_62/bias critic_network_7/dense_63/kernelcritic_network_7/dense_63/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� */
f*R(
&__inference_signature_wrapper_16878561

NoOpNoOp
� 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*� 
value�B� B�
�
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
�
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
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

kernel
bias*
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

kernel
bias*
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

kernel
bias*
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

kernel
bias*
O
6
_variables
7_iterations
8_learning_rate
9_update_step_xla*
* 

:serving_default* 
`Z
VARIABLE_VALUE critic_network_7/dense_60/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEcritic_network_7/dense_60/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE critic_network_7/dense_61/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEcritic_network_7/dense_61/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE critic_network_7/dense_62/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEcritic_network_7/dense_62/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE critic_network_7/dense_63/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEcritic_network_7/dense_63/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
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
�
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

@trace_0* 

Atrace_0* 

0
1*

0
1*
* 
�
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

Gtrace_0* 

Htrace_0* 

0
1*

0
1*
* 
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

Ntrace_0* 

Otrace_0* 

0
1*

0
1*
* 
�
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

Utrace_0* 

Vtrace_0* 

70*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
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
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename4critic_network_7/dense_60/kernel/Read/ReadVariableOp2critic_network_7/dense_60/bias/Read/ReadVariableOp4critic_network_7/dense_61/kernel/Read/ReadVariableOp2critic_network_7/dense_61/bias/Read/ReadVariableOp4critic_network_7/dense_62/kernel/Read/ReadVariableOp2critic_network_7/dense_62/bias/Read/ReadVariableOp4critic_network_7/dense_63/kernel/Read/ReadVariableOp2critic_network_7/dense_63/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpConst*
Tin
2	*
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
GPU2 *0J 8� **
f%R#
!__inference__traced_save_16878750
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename critic_network_7/dense_60/kernelcritic_network_7/dense_60/bias critic_network_7/dense_61/kernelcritic_network_7/dense_61/bias critic_network_7/dense_62/kernelcritic_network_7/dense_62/bias critic_network_7/dense_63/kernelcritic_network_7/dense_63/bias	iterationlearning_rate*
Tin
2*
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
GPU2 *0J 8� *-
f(R&
$__inference__traced_restore_16878790��
�
�
N__inference_critic_network_7_layer_call_and_return_conditional_losses_16878476	
state

action#
dense_60_16878420:R@
dense_60_16878422:@#
dense_61_16878437:@ 
dense_61_16878439: #
dense_62_16878454:  
dense_62_16878456: #
dense_63_16878470: 
dense_63_16878472:
identity�� dense_60/StatefulPartitionedCall� dense_61/StatefulPartitionedCall� dense_62/StatefulPartitionedCall� dense_63/StatefulPartitionedCallM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :r
concatConcatV2stateactionconcat/axis:output:0*
N*
T0*'
_output_shapes
:���������R�
 dense_60/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_60_16878420dense_60_16878422*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dense_60_layer_call_and_return_conditional_losses_16878419�
 dense_61/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0dense_61_16878437dense_61_16878439*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dense_61_layer_call_and_return_conditional_losses_16878436�
 dense_62/StatefulPartitionedCallStatefulPartitionedCall)dense_61/StatefulPartitionedCall:output:0dense_62_16878454dense_62_16878456*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dense_62_layer_call_and_return_conditional_losses_16878453�
 dense_63/StatefulPartitionedCallStatefulPartitionedCall)dense_62/StatefulPartitionedCall:output:0dense_63_16878470dense_63_16878472*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dense_63_layer_call_and_return_conditional_losses_16878469x
IdentityIdentity)dense_63/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_60/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall!^dense_62/StatefulPartitionedCall!^dense_63/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������>:���������: : : : : : : : 2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall:N J
'
_output_shapes
:���������>

_user_specified_namestate:OK
'
_output_shapes
:���������
 
_user_specified_nameaction
�/
�
$__inference__traced_restore_16878790
file_prefixC
1assignvariableop_critic_network_7_dense_60_kernel:R@?
1assignvariableop_1_critic_network_7_dense_60_bias:@E
3assignvariableop_2_critic_network_7_dense_61_kernel:@ ?
1assignvariableop_3_critic_network_7_dense_61_bias: E
3assignvariableop_4_critic_network_7_dense_62_kernel:  ?
1assignvariableop_5_critic_network_7_dense_62_bias: E
3assignvariableop_6_critic_network_7_dense_63_kernel: ?
1assignvariableop_7_critic_network_7_dense_63_bias:&
assignvariableop_8_iteration:	 *
 assignvariableop_9_learning_rate: 
identity_11��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp1assignvariableop_critic_network_7_dense_60_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp1assignvariableop_1_critic_network_7_dense_60_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp3assignvariableop_2_critic_network_7_dense_61_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp1assignvariableop_3_critic_network_7_dense_61_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp3assignvariableop_4_critic_network_7_dense_62_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp1assignvariableop_5_critic_network_7_dense_62_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp3assignvariableop_6_critic_network_7_dense_63_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp1assignvariableop_7_critic_network_7_dense_63_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_iterationIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_learning_rateIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
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
�!
�
!__inference__traced_save_16878750
file_prefix?
;savev2_critic_network_7_dense_60_kernel_read_readvariableop=
9savev2_critic_network_7_dense_60_bias_read_readvariableop?
;savev2_critic_network_7_dense_61_kernel_read_readvariableop=
9savev2_critic_network_7_dense_61_bias_read_readvariableop?
;savev2_critic_network_7_dense_62_kernel_read_readvariableop=
9savev2_critic_network_7_dense_62_bias_read_readvariableop?
;savev2_critic_network_7_dense_63_kernel_read_readvariableop=
9savev2_critic_network_7_dense_63_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0;savev2_critic_network_7_dense_60_kernel_read_readvariableop9savev2_critic_network_7_dense_60_bias_read_readvariableop;savev2_critic_network_7_dense_61_kernel_read_readvariableop9savev2_critic_network_7_dense_61_bias_read_readvariableop;savev2_critic_network_7_dense_62_kernel_read_readvariableop9savev2_critic_network_7_dense_62_bias_read_readvariableop;savev2_critic_network_7_dense_63_kernel_read_readvariableop9savev2_critic_network_7_dense_63_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
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

identity_1Identity_1:output:0*[
_input_shapesJ
H: :R@:@:@ : :  : : :: : : 2(
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
: :

_output_shapes
: 
�	
�
F__inference_dense_63_layer_call_and_return_conditional_losses_16878469

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
F__inference_dense_62_layer_call_and_return_conditional_losses_16878453

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
F__inference_dense_60_layer_call_and_return_conditional_losses_16878419

inputs0
matmul_readvariableop_resource:R@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:R@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������R: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������R
 
_user_specified_nameinputs
�	
�
F__inference_dense_63_layer_call_and_return_conditional_losses_16878696

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
+__inference_dense_62_layer_call_fn_16878666

inputs
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dense_62_layer_call_and_return_conditional_losses_16878453o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�&
�
N__inference_critic_network_7_layer_call_and_return_conditional_losses_16878617	
state

action9
'dense_60_matmul_readvariableop_resource:R@6
(dense_60_biasadd_readvariableop_resource:@9
'dense_61_matmul_readvariableop_resource:@ 6
(dense_61_biasadd_readvariableop_resource: 9
'dense_62_matmul_readvariableop_resource:  6
(dense_62_biasadd_readvariableop_resource: 9
'dense_63_matmul_readvariableop_resource: 6
(dense_63_biasadd_readvariableop_resource:
identity��dense_60/BiasAdd/ReadVariableOp�dense_60/MatMul/ReadVariableOp�dense_61/BiasAdd/ReadVariableOp�dense_61/MatMul/ReadVariableOp�dense_62/BiasAdd/ReadVariableOp�dense_62/MatMul/ReadVariableOp�dense_63/BiasAdd/ReadVariableOp�dense_63/MatMul/ReadVariableOpM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :r
concatConcatV2stateactionconcat/axis:output:0*
N*
T0*'
_output_shapes
:���������R�
dense_60/MatMul/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource*
_output_shapes

:R@*
dtype0�
dense_60/MatMulMatMulconcat:output:0&dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_60/BiasAdd/ReadVariableOpReadVariableOp(dense_60_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_60/BiasAddBiasAdddense_60/MatMul:product:0'dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_60/ReluReludense_60/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
dense_61/MatMulMatMuldense_60/Relu:activations:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_61/BiasAdd/ReadVariableOpReadVariableOp(dense_61_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_61/BiasAddBiasAdddense_61/MatMul:product:0'dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_61/ReluReludense_61/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_62/MatMul/ReadVariableOpReadVariableOp'dense_62_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_62/MatMulMatMuldense_61/Relu:activations:0&dense_62/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_62/BiasAdd/ReadVariableOpReadVariableOp(dense_62_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_62/BiasAddBiasAdddense_62/MatMul:product:0'dense_62/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_62/ReluReludense_62/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_63/MatMul/ReadVariableOpReadVariableOp'dense_63_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_63/MatMulMatMuldense_62/Relu:activations:0&dense_63/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_63/BiasAdd/ReadVariableOpReadVariableOp(dense_63_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_63/BiasAddBiasAdddense_63/MatMul:product:0'dense_63/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_63/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_60/BiasAdd/ReadVariableOp^dense_60/MatMul/ReadVariableOp ^dense_61/BiasAdd/ReadVariableOp^dense_61/MatMul/ReadVariableOp ^dense_62/BiasAdd/ReadVariableOp^dense_62/MatMul/ReadVariableOp ^dense_63/BiasAdd/ReadVariableOp^dense_63/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������>:���������: : : : : : : : 2B
dense_60/BiasAdd/ReadVariableOpdense_60/BiasAdd/ReadVariableOp2@
dense_60/MatMul/ReadVariableOpdense_60/MatMul/ReadVariableOp2B
dense_61/BiasAdd/ReadVariableOpdense_61/BiasAdd/ReadVariableOp2@
dense_61/MatMul/ReadVariableOpdense_61/MatMul/ReadVariableOp2B
dense_62/BiasAdd/ReadVariableOpdense_62/BiasAdd/ReadVariableOp2@
dense_62/MatMul/ReadVariableOpdense_62/MatMul/ReadVariableOp2B
dense_63/BiasAdd/ReadVariableOpdense_63/BiasAdd/ReadVariableOp2@
dense_63/MatMul/ReadVariableOpdense_63/MatMul/ReadVariableOp:N J
'
_output_shapes
:���������>

_user_specified_namestate:OK
'
_output_shapes
:���������
 
_user_specified_nameaction
�
�
+__inference_dense_60_layer_call_fn_16878626

inputs
unknown:R@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dense_60_layer_call_and_return_conditional_losses_16878419o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������R: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������R
 
_user_specified_nameinputs
�

�
F__inference_dense_62_layer_call_and_return_conditional_losses_16878677

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
F__inference_dense_61_layer_call_and_return_conditional_losses_16878436

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
F__inference_dense_60_layer_call_and_return_conditional_losses_16878637

inputs0
matmul_readvariableop_resource:R@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:R@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������R: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������R
 
_user_specified_nameinputs
�
�
+__inference_dense_63_layer_call_fn_16878686

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dense_63_layer_call_and_return_conditional_losses_16878469o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
3__inference_critic_network_7_layer_call_fn_16878583	
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
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallstateactionunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_critic_network_7_layer_call_and_return_conditional_losses_16878476o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������>:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:���������>

_user_specified_namestate:OK
'
_output_shapes
:���������
 
_user_specified_nameaction
�

�
F__inference_dense_61_layer_call_and_return_conditional_losses_16878657

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
+__inference_dense_61_layer_call_fn_16878646

inputs
unknown:@ 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dense_61_layer_call_and_return_conditional_losses_16878436o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�1
�
#__inference__wrapped_model_16878397

args_0

args_1J
8critic_network_7_dense_60_matmul_readvariableop_resource:R@G
9critic_network_7_dense_60_biasadd_readvariableop_resource:@J
8critic_network_7_dense_61_matmul_readvariableop_resource:@ G
9critic_network_7_dense_61_biasadd_readvariableop_resource: J
8critic_network_7_dense_62_matmul_readvariableop_resource:  G
9critic_network_7_dense_62_biasadd_readvariableop_resource: J
8critic_network_7_dense_63_matmul_readvariableop_resource: G
9critic_network_7_dense_63_biasadd_readvariableop_resource:
identity��0critic_network_7/dense_60/BiasAdd/ReadVariableOp�/critic_network_7/dense_60/MatMul/ReadVariableOp�0critic_network_7/dense_61/BiasAdd/ReadVariableOp�/critic_network_7/dense_61/MatMul/ReadVariableOp�0critic_network_7/dense_62/BiasAdd/ReadVariableOp�/critic_network_7/dense_62/MatMul/ReadVariableOp�0critic_network_7/dense_63/BiasAdd/ReadVariableOp�/critic_network_7/dense_63/MatMul/ReadVariableOp^
critic_network_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
critic_network_7/concatConcatV2args_0args_1%critic_network_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������R�
/critic_network_7/dense_60/MatMul/ReadVariableOpReadVariableOp8critic_network_7_dense_60_matmul_readvariableop_resource*
_output_shapes

:R@*
dtype0�
 critic_network_7/dense_60/MatMulMatMul critic_network_7/concat:output:07critic_network_7/dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
0critic_network_7/dense_60/BiasAdd/ReadVariableOpReadVariableOp9critic_network_7_dense_60_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
!critic_network_7/dense_60/BiasAddBiasAdd*critic_network_7/dense_60/MatMul:product:08critic_network_7/dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
critic_network_7/dense_60/ReluRelu*critic_network_7/dense_60/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
/critic_network_7/dense_61/MatMul/ReadVariableOpReadVariableOp8critic_network_7_dense_61_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
 critic_network_7/dense_61/MatMulMatMul,critic_network_7/dense_60/Relu:activations:07critic_network_7/dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
0critic_network_7/dense_61/BiasAdd/ReadVariableOpReadVariableOp9critic_network_7_dense_61_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
!critic_network_7/dense_61/BiasAddBiasAdd*critic_network_7/dense_61/MatMul:product:08critic_network_7/dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
critic_network_7/dense_61/ReluRelu*critic_network_7/dense_61/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
/critic_network_7/dense_62/MatMul/ReadVariableOpReadVariableOp8critic_network_7_dense_62_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
 critic_network_7/dense_62/MatMulMatMul,critic_network_7/dense_61/Relu:activations:07critic_network_7/dense_62/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
0critic_network_7/dense_62/BiasAdd/ReadVariableOpReadVariableOp9critic_network_7_dense_62_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
!critic_network_7/dense_62/BiasAddBiasAdd*critic_network_7/dense_62/MatMul:product:08critic_network_7/dense_62/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
critic_network_7/dense_62/ReluRelu*critic_network_7/dense_62/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
/critic_network_7/dense_63/MatMul/ReadVariableOpReadVariableOp8critic_network_7_dense_63_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
 critic_network_7/dense_63/MatMulMatMul,critic_network_7/dense_62/Relu:activations:07critic_network_7/dense_63/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
0critic_network_7/dense_63/BiasAdd/ReadVariableOpReadVariableOp9critic_network_7_dense_63_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!critic_network_7/dense_63/BiasAddBiasAdd*critic_network_7/dense_63/MatMul:product:08critic_network_7/dense_63/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������y
IdentityIdentity*critic_network_7/dense_63/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp1^critic_network_7/dense_60/BiasAdd/ReadVariableOp0^critic_network_7/dense_60/MatMul/ReadVariableOp1^critic_network_7/dense_61/BiasAdd/ReadVariableOp0^critic_network_7/dense_61/MatMul/ReadVariableOp1^critic_network_7/dense_62/BiasAdd/ReadVariableOp0^critic_network_7/dense_62/MatMul/ReadVariableOp1^critic_network_7/dense_63/BiasAdd/ReadVariableOp0^critic_network_7/dense_63/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������>:���������: : : : : : : : 2d
0critic_network_7/dense_60/BiasAdd/ReadVariableOp0critic_network_7/dense_60/BiasAdd/ReadVariableOp2b
/critic_network_7/dense_60/MatMul/ReadVariableOp/critic_network_7/dense_60/MatMul/ReadVariableOp2d
0critic_network_7/dense_61/BiasAdd/ReadVariableOp0critic_network_7/dense_61/BiasAdd/ReadVariableOp2b
/critic_network_7/dense_61/MatMul/ReadVariableOp/critic_network_7/dense_61/MatMul/ReadVariableOp2d
0critic_network_7/dense_62/BiasAdd/ReadVariableOp0critic_network_7/dense_62/BiasAdd/ReadVariableOp2b
/critic_network_7/dense_62/MatMul/ReadVariableOp/critic_network_7/dense_62/MatMul/ReadVariableOp2d
0critic_network_7/dense_63/BiasAdd/ReadVariableOp0critic_network_7/dense_63/BiasAdd/ReadVariableOp2b
/critic_network_7/dense_63/MatMul/ReadVariableOp/critic_network_7/dense_63/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������>
 
_user_specified_nameargs_0:OK
'
_output_shapes
:���������
 
_user_specified_nameargs_1
�

�
&__inference_signature_wrapper_16878561

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
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallargs_0args_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *,
f'R%
#__inference__wrapped_model_16878397o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������>:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������>
 
_user_specified_nameargs_0:OK
'
_output_shapes
:���������
 
_user_specified_nameargs_1"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
9
args_0/
serving_default_args_0:0���������>
9
args_1/
serving_default_args_1:0���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:�b
�
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
�
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
�
trace_02�
3__inference_critic_network_7_layer_call_fn_16878583�
���
FullArgSpec&
args�
jself
jstate
jaction
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�
trace_02�
N__inference_critic_network_7_layer_call_and_return_conditional_losses_16878617�
���
FullArgSpec&
args�
jself
jstate
jaction
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�B�
#__inference__wrapped_model_16878397args_0args_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
j
6
_variables
7_iterations
8_learning_rate
9_update_step_xla"
experimentalOptimizer
 "
trackable_dict_wrapper
,
:serving_default"
signature_map
2:0R@2 critic_network_7/dense_60/kernel
,:*@2critic_network_7/dense_60/bias
2:0@ 2 critic_network_7/dense_61/kernel
,:* 2critic_network_7/dense_61/bias
2:0  2 critic_network_7/dense_62/kernel
,:* 2critic_network_7/dense_62/bias
2:0 2 critic_network_7/dense_63/kernel
,:*2critic_network_7/dense_63/bias
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
�B�
3__inference_critic_network_7_layer_call_fn_16878583stateaction"�
���
FullArgSpec&
args�
jself
jstate
jaction
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_critic_network_7_layer_call_and_return_conditional_losses_16878617stateaction"�
���
FullArgSpec&
args�
jself
jstate
jaction
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
@trace_02�
+__inference_dense_60_layer_call_fn_16878626�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z@trace_0
�
Atrace_02�
F__inference_dense_60_layer_call_and_return_conditional_losses_16878637�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zAtrace_0
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
�
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
�
Gtrace_02�
+__inference_dense_61_layer_call_fn_16878646�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zGtrace_0
�
Htrace_02�
F__inference_dense_61_layer_call_and_return_conditional_losses_16878657�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zHtrace_0
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
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�
Ntrace_02�
+__inference_dense_62_layer_call_fn_16878666�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zNtrace_0
�
Otrace_02�
F__inference_dense_62_layer_call_and_return_conditional_losses_16878677�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zOtrace_0
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
�
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�
Utrace_02�
+__inference_dense_63_layer_call_fn_16878686�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zUtrace_0
�
Vtrace_02�
F__inference_dense_63_layer_call_and_return_conditional_losses_16878696�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zVtrace_0
'
70"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
�2��
���
FullArgSpec2
args*�'
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
&__inference_signature_wrapper_16878561args_0args_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_dense_60_layer_call_fn_16878626inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_60_layer_call_and_return_conditional_losses_16878637inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_dense_61_layer_call_fn_16878646inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_61_layer_call_and_return_conditional_losses_16878657inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_dense_62_layer_call_fn_16878666inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_62_layer_call_and_return_conditional_losses_16878677inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_dense_63_layer_call_fn_16878686inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_63_layer_call_and_return_conditional_losses_16878696inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
#__inference__wrapped_model_16878397�Q�N
G�D
 �
args_0���������>
 �
args_1���������
� "3�0
.
output_1"�
output_1����������
N__inference_critic_network_7_layer_call_and_return_conditional_losses_16878617�P�M
F�C
�
state���������>
 �
action���������
� ",�)
"�
tensor_0���������
� �
3__inference_critic_network_7_layer_call_fn_16878583P�M
F�C
�
state���������>
 �
action���������
� "!�
unknown����������
F__inference_dense_60_layer_call_and_return_conditional_losses_16878637c/�,
%�"
 �
inputs���������R
� ",�)
"�
tensor_0���������@
� �
+__inference_dense_60_layer_call_fn_16878626X/�,
%�"
 �
inputs���������R
� "!�
unknown���������@�
F__inference_dense_61_layer_call_and_return_conditional_losses_16878657c/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0��������� 
� �
+__inference_dense_61_layer_call_fn_16878646X/�,
%�"
 �
inputs���������@
� "!�
unknown��������� �
F__inference_dense_62_layer_call_and_return_conditional_losses_16878677c/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0��������� 
� �
+__inference_dense_62_layer_call_fn_16878666X/�,
%�"
 �
inputs��������� 
� "!�
unknown��������� �
F__inference_dense_63_layer_call_and_return_conditional_losses_16878696c/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������
� �
+__inference_dense_63_layer_call_fn_16878686X/�,
%�"
 �
inputs��������� 
� "!�
unknown����������
&__inference_signature_wrapper_16878561�e�b
� 
[�X
*
args_0 �
args_0���������>
*
args_1 �
args_1���������"3�0
.
output_1"�
output_1���������