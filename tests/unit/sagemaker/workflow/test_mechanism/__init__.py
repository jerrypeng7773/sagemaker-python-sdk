# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import os

from mock import Mock, PropertyMock
from sagemaker import TrainingInput
from sagemaker.debugger import Rule, DebuggerHookConfig, TensorBoardOutputConfig, ProfilerConfig
from sagemaker.huggingface.estimator import TrainingCompilerConfig
from sagemaker.clarify import DataConfig
from sagemaker.network import NetworkConfig
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.amazon.amazon_estimator import RecordSet
from sagemaker.rl.estimator import RLToolkit, RLFramework
from sagemaker.pytorch import PyTorch
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import (
    ParameterString,
    ParameterInteger,
    ParameterFloat,
    ParameterBoolean,
)
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile, Properties
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, TuningStep, TransformStep
from tests.unit import DATA_DIR

STR_VAL = "MyString"
ROLE = "DummyRole"
INSTANCE_TYPE = "ml.m5.xlarge"
BUCKET = "my-bucket"
REGION = "us-west-2"
DUMMY_S3_SCRIPT_PATH = "s3://dummy-s3/dummy_script.py"
TENSORFLOW_PATH = os.path.join(DATA_DIR, "tfs/tfs-test-entrypoint-and-dependencies")
TENSORFLOW_ENTRY_POINT = os.path.join(TENSORFLOW_PATH, "inference.py")


def generate_mock_pipeline_session():
    """Generate mock pipeline session"""
    client_mock = Mock()
    client_mock._client_config.user_agent = (
        "Boto3/1.14.24 Python/3.8.5 Linux/5.4.0-42-generic Botocore/1.17.24 Resource"
    )
    client_mock.describe_algorithm.return_value = {
        "TrainingSpecification": {
            "TrainingChannels": [
                {
                    "SupportedContentTypes": ["application/json"],
                    "SupportedInputModes": ["File"],
                    "Name": "train",
                }
            ],
            "SupportedTrainingInstanceTypes": ["ml.m5.xlarge", "ml.m4.xlarge"],
            "SupportedHyperParameters": [
                {
                    "Name": "MyKey",
                    "Type": "FreeText",
                }
            ],
        },
        "AlgorithmName": "algo-name",
    }

    role_mock = Mock()
    type(role_mock).arn = PropertyMock(return_value=ROLE)
    resource_mock = Mock()
    resource_mock.Role.return_value = role_mock
    session_mock = Mock(region_name=REGION)
    session_mock.resource.return_value = resource_mock
    session_mock.client.return_value = client_mock

    return PipelineSession(
        boto_session=session_mock,
        sagemaker_client=client_mock,
        default_bucket=BUCKET,
    )


def _generate_all_pipeline_vars() -> dict:
    """Generate a dic with all kinds of Pipeline variables"""
    # Parameter
    ppl_param_str = ParameterString(name="MyString")
    ppl_param_int = ParameterInteger(name="MyInt")
    ppl_param_float = ParameterFloat(name="MyFloat")
    ppl_param_bool = ParameterBoolean(name="MyBool")

    # Function
    ppl_join = Join(on=" ", values=[ppl_param_int, ppl_param_float])
    property_file = PropertyFile(
        name="name",
        output_name="result",
        path="output",
    )
    ppl_json_get = JsonGet(
        step_name="my-step",
        property_file=property_file,
        json_path="my-json-path",
    )

    # Properties
    ppl_prop = Properties("Steps.MyStep", "DescribeTrainingJobResponse")
    ppl_prop_str = ppl_prop.OutputDataConfig.S3OutputPath
    ppl_prop_int = ppl_prop.TrainingTimeInSeconds
    ppl_prop_float = ppl_prop.FinalMetricDataList[0].Value
    ppl_prop_bool = ppl_prop.EnableManagedSpotTraining

    # Execution Variables
    ppl_exe_var = ExecutionVariables.PIPELINE_NAME

    return dict(
        str=[
            (
                ppl_param_str,
                dict(origin=ppl_param_str.expr, to_string=ppl_param_str.to_string().expr),
            ),
            (ppl_join, dict(origin=ppl_join.expr, to_string=ppl_join.to_string().expr)),
            (ppl_json_get, dict(origin=ppl_json_get.expr, to_string=ppl_json_get.to_string().expr)),
            (ppl_prop_str, dict(origin=ppl_prop_str.expr, to_string=ppl_prop_str.to_string().expr)),
            (ppl_exe_var, dict(origin=ppl_exe_var.expr, to_string=ppl_exe_var.to_string().expr)),
        ],
        int=[
            (
                ppl_param_int,
                dict(origin=ppl_param_int.expr, to_string=ppl_param_int.to_string().expr),
            ),
            (ppl_json_get, dict(origin=ppl_json_get.expr, to_string=ppl_json_get.to_string().expr)),
            (ppl_prop_int, dict(origin=ppl_prop_int.expr, to_string=ppl_prop_int.to_string().expr)),
        ],
        float=[
            (
                ppl_param_float,
                dict(origin=ppl_param_float.expr, to_string=ppl_param_float.to_string().expr),
            ),
            (ppl_json_get, dict(origin=ppl_json_get.expr, to_string=ppl_json_get.to_string().expr)),
            (
                ppl_prop_float,
                dict(origin=ppl_prop_float.expr, to_string=ppl_prop_float.to_string().expr),
            ),
        ],
        bool=[
            (
                ppl_param_bool,
                dict(origin=ppl_param_bool.expr, to_string=ppl_param_bool.to_string().expr),
            ),
            (ppl_json_get, dict(origin=ppl_json_get.expr, to_string=ppl_json_get.to_string().expr)),
            (
                ppl_prop_bool,
                dict(origin=ppl_prop_bool.expr, to_string=ppl_prop_bool.to_string().expr),
            ),
        ],
    )


PIPELINE_SESSION = generate_mock_pipeline_session()
PIPELINE_VARIABLES = _generate_all_pipeline_vars()
FIXED_ARGUMENTS = dict(
    common=dict(
        instance_type=INSTANCE_TYPE,
        role=ROLE,
        sagemaker_session=PIPELINE_SESSION,
        source_dir=f"s3://{BUCKET}/source",
        entry_point=TENSORFLOW_ENTRY_POINT,
        dependencies=[os.path.join(TENSORFLOW_PATH, "dependency.py")],
        code_location=f"s3://{BUCKET}/code",
    ),
    processor=dict(
        estimator_cls=PyTorch,
        code=f"s3://{BUCKET}/code",
        spark_event_logs_s3_uri=f"s3://{BUCKET}/my-spark-output-path",
        framework_version="1.8",
        network_config=NetworkConfig(
            subnets=[ParameterString(name="nw_cfg_subnets")],
            security_group_ids=[ParameterString(name="nw_cfg_security_group_ids")],
            enable_network_isolation=ParameterBoolean(name="nw_cfg_enable_nw_isolation"),
            encrypt_inter_container_traffic=ParameterBoolean(
                name="nw_cfg_encrypt_inter_container_traffic"
            ),
        ),  # TODO: need to recursively assign with Pipeline Variable in later changes
        inputs=[
            ProcessingInput(
                source=ParameterString(name="proc_input_source"),
                destination=ParameterString(name="proc_input_dest"),
                s3_data_type=ParameterString(name="proc_input_s3_data_type"),
                app_managed=ParameterBoolean(name="proc_input_app_managed"),
            ),  # TODO: need to recursively assign with Pipeline Variable in later changes
        ],
        outputs=[
            ProcessingOutput(
                source=ParameterString(name="proc_output_source"),
                destination=ParameterString(name="proc_output_dest"),
                app_managed=ParameterBoolean(name="proc_output_app_managed"),
            ),  # TODO: need to recursively assign with Pipeline Variable in later changes
        ],
        data_config=DataConfig(
            s3_data_input_path=ParameterString(name="clarify_processor_input"),
            s3_output_path=ParameterString(name="clarify_processor_output"),
            s3_analysis_config_output_path="s3://analysis_config_output_path",
        ),
        data_bias_config=DataConfig(
            s3_data_input_path=ParameterString(name="clarify_processor_input"),
            s3_output_path=ParameterString(name="clarify_processor_output"),
            s3_analysis_config_output_path="s3://analysis_config_output_path",
        ),
    ),
    estimator=dict(
        image_uri_region="us-west-2",
        input_mode="File",
        compiler_config=TrainingCompilerConfig(enabled=False),
        records=RecordSet(
            s3_data=ParameterString("records_s3_data"),
            num_records=1000,
            feature_dim=128,
            channel="training",
        ),
        vector_dim=128,
        enc_dim=128,
        momentum=1e-6,
        beta_1=1e-4,
        beta_2=1e-4,
        mini_batch_size=1000,
        dropout=0.25,
        num_classes=10,
        mlp_dim=512,
        mlp_activation="relu",
        output_layer="softmax",
        comparator_list="hadamard,concat,abs_diff",
        token_embedding_storage_type="dense",
        enc0_network="bilstm",
        enc1_network="bilstm",
        enc0_token_embedding_dim=256,
        enc1_token_embedding_dim=256,
        enc0_vocab_size=512,
        enc1_vocab_size=512,
        bias_init_method="normal",
        factors_init_method="normal",
        predictor_type="regressor",
        linear_init_method="uniform",
        toolkit=RLToolkit.RAY,
        toolkit_version="1.6.0",
        framework=RLFramework.PYTORCH,
        algorithm_mode="regular",
        num_topics=6,
        k=6,
        init_method="kmeans++",
        local_init_method="kmeans++",
        eval_metrics="mds,ssd",
        tol=1e-4,
        dimension_reduction_type="sign",
        index_type="faiss.Flat",
        faiss_index_ivf_nlists="auto",
        index_metric="COSINE",
        binary_classifier_model_selection_criteria="f1",
        positive_example_weight_mult="auto",
        loss="logistic",
        target_recall=0.1,
        target_precision=0.8,
        early_stopping_tolerance=1e-4,
        encoder_layers_activation="relu",
        optimizer="adam",
        tolerance=1e-4,
        rescale_gradient=1e-2,
        weight_decay=1e-6,
        learning_rate=1e-4,
        num_trees=50,
        source_dir=f"s3://{BUCKET}/source",
        entry_point=os.path.join(TENSORFLOW_PATH, "inference.py"),
        dependencies=[os.path.join(TENSORFLOW_PATH, "dependency.py")],
        code_location=f"s3://{BUCKET}/code",
        output_path=f"s3://{BUCKET}/output",
        model_uri=f"s3://{BUCKET}/model",
        py_version="py2",
        framework_version="2.1.1",
        rules=[
            Rule.custom(  # TODO: need to recursively assign with Pipeline Variable in later changes
                name="CustomRule",
                image_uri="RuleImageUri",  # TODO check why if it's pipeline var, profiler_config is ignored
                instance_type=ParameterString(name="rules_instance_type"),
                volume_size_in_gb=5,  # TODO check why if it's pipeline var, profiler_config is ignored
                source="path/to/my_custom_rule.py",
                rule_to_invoke="CustomRule",
                other_trials_s3_input_paths=["s3://path/trial1", "s3://path/trial2"],
                rule_parameters={"threshold": "120"},
            )
        ],
        debugger_hook_config=DebuggerHookConfig(
            s3_output_path=ParameterString(name="debugger_hook_s3_output")
        ),  # TODO: need to recursively assign with Pipeline Variable in later changes
        tensorboard_output_config=TensorBoardOutputConfig(
            s3_output_path=ParameterString(name="tensorboard_s3_output")
        ),  # TODO: need to recursively assign with Pipeline Variable in later changes
        profiler_config=ProfilerConfig(
            system_monitor_interval_millis=ParameterInteger(name="profile_config_system_monitor")
        ),  # TODO: need to recursively assign with Pipeline Variable in later changes
        inputs={
            "train": TrainingInput(
                s3_data=ParameterString(name="train_inputs_s3_data"),
                content_type=ParameterString(name="train_inputs_content_type"),
            ),
        },
    ),
    transformer=dict(),
    tuner=dict(),
    model=dict(),
)
STEP_CLASS = dict(
    processor=ProcessingStep,
    estimator=TrainingStep,
    transformer=TransformStep,
    tuner=TuningStep,
    model=ModelStep,
)

# A dict for class __init__ parameters which are not used in
# request dict generated by a specific target function but are used in another
# For example, as for Model class constructor, the vpc_config is used only in
# model.create and is ignored in model.register
# Thus when testing on model.register we should skip replacing vpc_config
# with pipeline variables
CLASS_PARAMS_EXCLUDED_IN_TARGET_FUNC = dict(
    model=dict(register={"vpc_config", "enable_network_isolation"}),
)
# A dict for base class __init__ parameters which are not used in
# some specific subclasses.
# For example, TensorFlowModel uses **kwarg for duplicate parameters
# in base class (FrameworkModel/Model) but it ignores the "image_config"
# in target functions.
BASE_CLASS_PARAMS_EXCLUDED_IN_SUB_CLASS = dict(
    model=dict(TensorFlowModel={"image_config"}),
    estimator=dict(
        AlgorithmEstimator={
            "enable_network_isolation",
            "container_log_level",
            "checkpoint_s3_uri",
            "checkpoint_local_path",
            "enable_sagemaker_metrics",
            "environment",
            "max_retry_attempts",
        },
        SKLearn={
            "instance_count",
            "instance_type",
        },
    ),
)
# A dict to keep the optional arguments which should not be None according to the logic
# E.g. model_channel_name should not be None if model_uri is presented
# Given the test mechanism only one arg to be None in each round, this can cause error.
# Thus we can skip it
PARAMS_SHOULD_NOT_BE_NONE = dict(
    estimator=dict(init={"instance_count", "instance_type", "model_channel_name"}),
    model=dict(register={"model_data"}, create={"role"}),
)
# A dict for parameters which should not be replaced with pipeline variables
# since they are bonded with other parameters with None value. For example:
# Case 1: if outputs (a parameter in FrameworkProcessor.run) is None,
# output_kms_key (a parameter in constructor) is omitted
# so don't need to replace it with pipeline variables
# Case 2: if image_uri is None, instance_type is not allowed to be pipeline variables,
# otherwise, the the class can fail to be initiated
PARAM_BONDED_WITH_ANOTHER = dict(
    processor=dict(
        init=dict(instance_type={"image_uri"}),
        run=dict(output_kms_key={"outputs"}),
    ),
    estimator=dict(
        fit=dict(
            subnets={"security_group_ids"},
            security_group_ids={"subnets"},
            model_channel_name={"model_uri"},
            checkpoint_local_path={"checkpoint_s3_uri"},
            instance_type={"image_uri"},
        ),
    ),
    model=dict(
        register=dict(
            env={"model_package_group_name"},
            image_config={"model_package_group_name"},
        ),
    ),
)
# A dict for mutual excluded parameters
# For example, if model_package_name is not None, model_package_group_name should be None
# and should skip to be replaced with a pipeline variable
MUTUAL_EXCLUDED_PARAMS = dict(
    model=dict(
        register=dict(
            model_package_name={"model_package_group_name"},
            model_package_group_name={"model_package_name"},
        ),
    )
)
