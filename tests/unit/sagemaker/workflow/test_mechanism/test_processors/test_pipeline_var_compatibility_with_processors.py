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

# from sagemaker.processing import FrameworkProcessor, ScriptProcessor, Processor
# from sagemaker.pytorch.processing import PyTorchProcessor
# from sagemaker.clarify import SageMakerClarifyProcessor
# from sagemaker.tensorflow.processing import TensorFlowProcessor
# from sagemaker.xgboost.processing import XGBoostProcessor
from sagemaker.spark.processing import SparkJarProcessor, PySparkProcessor
from sagemaker.mxnet.processing import MXNetProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.huggingface.processing import HuggingFaceProcessor
from tests.unit.sagemaker.workflow.test_mechanism.test_pipeline_var_compatibility_template import (
    PipelineVarCompatiTestTemplate,
)
from tests.unit.sagemaker.workflow.test_mechanism import (
    ROLE,
    DUMMY_S3_SCRIPT_PATH,
    PIPELINE_SESSION,
)


# These tests provide the incomplete default arg dict
# within which some class or target func parameters are missing or assigned a None value.
# The test template will fill in those missing/None args
# Note: the default args should not include PipelineVariable objects
# def test_processor_compatibility():
#     default_args = dict(
#         clazz_args=dict(
#             role=ROLE,
#             volume_size_in_gb=None,
#             sagemaker_session=PIPELINE_SESSION,
#         ),
#         func_args=dict(),
#     )
#     test_template = PipelineVarCompatiTestTemplate(
#         clazz=Processor,
#         default_args=default_args,
#     )
#     test_template.check_compatibility()
#
#
# def test_script_processor_compatibility():
#     default_args = dict(
#         clazz_args=dict(
#             role=ROLE,
#             volume_size_in_gb=None,
#             sagemaker_session=PIPELINE_SESSION,
#         ),
#         func_args=dict(
#             code=DUMMY_S3_SCRIPT_PATH,
#         ),
#     )
#     test_template = PipelineVarCompatiTestTemplate(
#         clazz=ScriptProcessor,
#         default_args=default_args,
#     )
#     test_template.check_compatibility()
#
#
# def test_framework_processor_compatibility():
#     default_args = dict(
#         clazz_args=dict(
#             estimator_cls=None,
#             role=ROLE,
#             py_version="py3",
#             volume_size_in_gb=None,
#             sagemaker_session=PIPELINE_SESSION,
#         ),
#         func_args=dict(
#             code=DUMMY_S3_SCRIPT_PATH,
#         ),
#     )
#     test_template = PipelineVarCompatiTestTemplate(
#         clazz=FrameworkProcessor,
#         default_args=default_args,
#     )
#     test_template.check_compatibility()

# def test_pytorch_processor_compatibility():
#     default_args = dict(
#         clazz_args=dict(
#             framework_version="1.8.1",
#             role=ROLE,
#             py_version="py3",
#             volume_size_in_gb=None,
#             sagemaker_session=PIPELINE_SESSION,
#         ),
#         func_args=dict(
#             code=DUMMY_S3_SCRIPT_PATH,
#         ),
#     )
#     test_template = PipelineVarCompatiTestTemplate(
#         clazz=PyTorchProcessor,
#         default_args=default_args,
#     )
#     test_template.check_compatibility()


# def test_sagemaker_clarify_processor():
#     default_args = dict(
#         clazz_args=dict(
#             role=ROLE,
#             sagemaker_session=PIPELINE_SESSION,
#         ),
#         func_args=dict(
#             code=DUMMY_S3_SCRIPT_PATH,
#         ),
#     )
#     test_template = PipelineVarCompatiTestTemplate(
#         clazz=SageMakerClarifyProcessor,
#         default_args=default_args,
#     )
#     test_template.check_compatibility()

# def test_tensorflow_processor():
#     default_args = dict(
#         clazz_args=dict(
#             framework_version="2.8",
#             role=ROLE,
#             py_version="py39",
#             sagemaker_session=PIPELINE_SESSION,
#         ),
#         func_args=dict(
#             code=DUMMY_S3_SCRIPT_PATH,
#         ),
#     )
#     test_template = PipelineVarCompatiTestTemplate(
#         clazz=TensorFlowProcessor,
#         default_args=default_args,
#     )
#     test_template.check_compatibility()


# def test_xgboost_processor():
#     default_args = dict(
#         clazz_args=dict(
#             role=ROLE,
#             framework_version="1.2-1",
#             py_version="py3",
#             sagemaker_session=PIPELINE_SESSION,
#         ),
#         func_args=dict(
#             code=DUMMY_S3_SCRIPT_PATH,
#         ),
#     )
#     test_template = PipelineVarCompatiTestTemplate(
#         clazz=XGBoostProcessor,
#         default_args=default_args,
#     )
#     test_template.check_compatibility()

# def test_spark_jar_processor():
#     # takes a really long time, since the .run has many args
#     default_args = dict(
#             clazz_args=dict(
#                 role=ROLE,
#                 framework_version="2.4",
#                 py_version="py37",
#                 sagemaker_session=PIPELINE_SESSION,
#             ),
#             func_args=dict(
#                 submit_app=DUMMY_S3_SCRIPT_PATH,
#             ),
#         )
#     test_template = PipelineVarCompatiTestTemplate(
#         clazz=SparkJarProcessor,
#         default_args=default_args,
#     )
#     test_template.check_compatibility()


# def test_py_spark_processor():
#     # takes a really long time, since the .run has many args
#     default_args = dict(
#             clazz_args=dict(
#                 role=ROLE,
#                 framework_version="2.4",
#                 py_version="py37",
#                 sagemaker_session=PIPELINE_SESSION,
#             ),
#             func_args=dict(
#                 submit_app=DUMMY_S3_SCRIPT_PATH,
#             ),
#         )
#     test_template = PipelineVarCompatiTestTemplate(
#         clazz=PySparkProcessor,
#         default_args=default_args,
#     )
#     test_template.check_compatibility()


# def test_mxnet_processor():
#     # takes a really long time, since the .run has many args
#     default_args = dict(
#             clazz_args=dict(
#                 role=ROLE,
#                 framework_version="1.6",
#                 py_version="py3",
#                 sagemaker_session=PIPELINE_SESSION,
#             ),
#             func_args=dict(
#                 code=DUMMY_S3_SCRIPT_PATH,
#             ),
#         )
#     test_template = PipelineVarCompatiTestTemplate(
#         clazz=MXNetProcessor,
#         default_args=default_args,
#     )
#     test_template.check_compatibility()

# def test_sklearn_processor():
#     default_args = dict(
#             clazz_args=dict(
#                 role=ROLE,
#                 framework_version="0.23-1",
#                 sagemaker_session=PIPELINE_SESSION,
#                 instance_type="ml.m5.xlarge",
#             ),
#             func_args=dict(
#                 code=DUMMY_S3_SCRIPT_PATH,
#             ),
#         )
#     test_template = PipelineVarCompatiTestTemplate(
#         clazz=SKLearnProcessor,
#         default_args=default_args,
#     )
#     test_template.check_compatibility()

# def test_hugging_face_processor():
#     default_args = dict(
#         clazz_args=dict(
#             role=ROLE,
#             sagemaker_session=PIPELINE_SESSION,
#             pytorch_version=None,
#         ),
#         func_args=dict(
#             code=DUMMY_S3_SCRIPT_PATH,
#         ),
#     )
#     test_template = PipelineVarCompatiTestTemplate(
#         clazz=HuggingFaceProcessor,
#         default_args=default_args,
#     )
#     test_template.check_compatibility()