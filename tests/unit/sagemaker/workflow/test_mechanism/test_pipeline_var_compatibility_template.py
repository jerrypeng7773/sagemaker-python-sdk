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

import json

from random import getrandbits
from typing import Optional
from typing_extensions import get_origin

from sagemaker import Model, PipelineModel
from sagemaker.estimator import EstimatorBase, Estimator
from sagemaker.amazon.amazon_estimator import AmazonAlgorithmEstimatorBase
from sagemaker.processing import Processor
from sagemaker.clarify import SageMakerClarifyProcessor
from sagemaker.transformer import Transformer
from sagemaker.tuner import HyperparameterTuner
from sagemaker.workflow.pipeline import Pipeline
from tests.unit.sagemaker.workflow.test_mechanism import (
    STEP_CLASS,
    FIXED_ARGUMENTS,
    STR_VAL,
    PARAMS_SHOULD_NOT_BE_NONE,
    PIPELINE_SESSION,
    CLASS_PARAMS_EXCLUDED_IN_TARGET_FUNC,
    PARAM_BONDED_WITH_ANOTHER,
    MUTUAL_EXCLUDED_PARAMS,
    BASE_CLASS_PARAMS_EXCLUDED_IN_SUB_CLASS,
)
from tests.unit.sagemaker.workflow.test_mechanism.utilities import (
    support_pipeline_variable,
    get_param_dict,
    generate_pipeline_vars_per_type,
    clean_up_types,
)


class PipelineVarCompatiTestTemplate:
    """Check the compatibility between Pipeline variables and the given class, target method"""

    def __init__(self, clazz: type, default_args: dict):
        """Initialize a `PipelineVarCompatiTestTemplate` instance.

        Args:
            clazz (type): The class to test the compatibility.
            default_args (dict): The given default arguments for the class and its target method.
        """
        self._clazz = clazz
        self._clazz_type = self._get_clazz_type()
        self._target_funcs = self._get_target_functions()
        self._clz_params = get_param_dict(clazz.__init__, clazz)
        self._func_params = dict()
        for func in self._target_funcs:
            self._func_params[func.__name__] = get_param_dict(func)
        self._set_and_restructure_default_args(default_args)

    def _set_and_restructure_default_args(self, default_args: dict):
        """Set and restructure the default_args

        Restructure the default_args["func_args"] if it's missing the layer of target function name

        Args:
            default_args (dict): The given default arguments for the class and its target method.
        """
        self._default_args = default_args
        # restructure the default_args["func_args"] if it's missing the layer of target function name
        if len(self._target_funcs) == 1:
            target_func_name = self._target_funcs[0].__name__
            if target_func_name not in default_args["func_args"]:
                args = self._default_args.pop("func_args")
                self._default_args["func_args"] = dict()
                self._default_args["func_args"][target_func_name] = args

        self._check_or_fill_in_args(
            params={**self._clz_params["required"], **self._clz_params["optional"]},
            default_args=self._default_args["clazz_args"],
        )
        for func in self._target_funcs:
            func_name = func.__name__
            self._check_or_fill_in_args(
                params={
                    **self._func_params[func_name]["required"],
                    **self._func_params[func_name]["optional"],
                },
                default_args=self._default_args["func_args"][func_name],
            )
        print("default_args", self._default_args)

    def _get_clazz_type(self) -> str:
        """Get the type (in str) of the downstream class"""
        if issubclass(self._clazz, Processor):
            return "processor"
        if issubclass(self._clazz, EstimatorBase):
            return "estimator"
        if issubclass(self._clazz, Transformer):
            return "transformer"
        if issubclass(self._clazz, HyperparameterTuner):
            return "tuner"
        if issubclass(self._clazz, (Model, PipelineModel)):
            return "model"
        raise TypeError(f"Unsupported downstream class: {self._clazz}")

    def check_compatibility(self):
        """The entry to check the compatibility"""
        print(
            "Starting to check Pipeline variable compatibility for class (%s) and target methods (%s)\n"
            % (self._clazz.__name__, [func.__name__ for func in self._target_funcs])
        )

        # Check the case when all args are assigned not-None values
        print("## Starting to check the compatibility when all optional args are not None ##")
        self._iterate_params_to_check_compatibility()

        # Check the case when one of the optional arg is None
        print(
            "## Starting to check the compatibility when one of the optional arg is None in each round ##"
        )
        self._iterate_optional_params_to_check_compatibility()

    def _iterate_params_to_check_compatibility(
        self,
        param_with_none: Optional[str] = None,
        test_func_for_none: Optional[str] = None,
    ):
        """Iterate each parameter and assign a pipeline var to it to test compatibility

        Args:
            param_with_none (str): The name of the parameter with None value.
            test_func_for_none (str): The name of the function which is being tested by
                replacing optional parameters to None.
        """
        self._iterate_clz_params_to_check_compatibility(param_with_none, test_func_for_none)
        self._iterate_func_params_to_check_compatibility(param_with_none, test_func_for_none)

    def _iterate_optional_params_to_check_compatibility(self):
        """Iterate each optional parameter and set it to none to test compatibility"""
        self._iterate_class_optional_params()
        self._iterate_func_optional_params()

    def _iterate_class_optional_params(self):
        """Iterate each optional parameter in class __init__ and check compatibility"""
        print("### Starting to iterate optional parameters in class __init__")
        self._iterate_optional_params(
            optional_params=self._clz_params["optional"],
            default_args=self._default_args["clazz_args"],
        )

    def _iterate_func_optional_params(self):
        """Iterate each function parameter and check compatibility"""
        for func in self._target_funcs:
            print(f"### Starting to iterate optional parameters in function {func.__name__}")
            self._iterate_optional_params(
                optional_params=self._func_params[func.__name__]["optional"],
                default_args=self._default_args["func_args"][func.__name__],
                test_func_for_none=func.__name__,
            )

    def _iterate_optional_params(
        self,
        optional_params: dict,
        default_args: dict,
        test_func_for_none: Optional[str] = None,
    ):
        """Iterate each optional parameter and check compatibility
        Args:
            optional_params (dict): The dict containing the optional parameters of a class or method.
            default_args (dict): The dict containing the default arguments of a class or method.
            test_func_for_none (str): The name of the function which is being tested by
                replacing optional parameters to None.
        """
        for param_name in optional_params.keys():
            if self._skip_test_on_param_should_not_be_none(param_name, "init"):
                continue
            origin_val = default_args[param_name]
            default_args[param_name] = None
            print("=== Parameter (%s) is None in this round ===" % param_name)
            self._iterate_params_to_check_compatibility(param_name, test_func_for_none)
            default_args[param_name] = origin_val

    def _iterate_clz_params_to_check_compatibility(
        self,
        param_with_none: Optional[str] = None,
        test_func_for_none: Optional[str] = None,
    ):
        """Iterate each class parameter and assign a pipeline var to it to test compatibility

        Args:
            param_with_none (str): The name of the parameter with None value.
            test_func_for_none (str): The name of the function which is being tested by
                replacing optional parameters to None.
        """
        print(
            f"#### Iterating parameters (supporting PipelineVariable) in class {self._clazz.__name__} __init__ function"
        )
        clz_params = {**self._clz_params["required"], **self._clz_params["optional"]}
        # Iterate through each default arg
        for clz_param_name, clz_default_arg in self._default_args["clazz_args"].items():
            if clz_param_name == param_with_none:
                continue
            clz_param_type = clz_params[clz_param_name]["type"]
            if not support_pipeline_variable(clz_param_type):
                continue
            if self._skip_test_on_bonded_params(
                param_name=clz_param_name,
                param_with_none=param_with_none,
                target_func="init",
            ):
                continue
            if self._skip_test_on_base_clz_param_excluded_in_subclz(clz_param_name):
                continue

            # For each arg which supports pipeline variables,
            # Replace it with each one of generated pipeline variables
            ppl_vars = generate_pipeline_vars_per_type(clz_param_name, clz_param_type)
            for clz_ppl_var, expected_clz_expr in ppl_vars:
                print(
                    "Replacing class arg (%s) with pipeline variable which is expected to be (%s)"
                    % (clz_param_name, expected_clz_expr)
                )
                self._default_args["clazz_args"][clz_param_name] = clz_ppl_var

                obj = self._clazz(**self._default_args["clazz_args"])
                for func in self._target_funcs:
                    func_name = func.__name__
                    if test_func_for_none and test_func_for_none != func_name:
                        # Iterating optional parameters of a specific target function,
                        # which does not impact other target functions,
                        # so we can skip them
                        continue
                    if self._skip_test_on_bonded_params(
                        param_name=clz_param_name,
                        param_with_none=param_with_none,
                        target_func=func_name,
                    ):
                        continue
                    if self._skip_test_on_mutual_exclusive_params(clz_param_name, func_name):
                        continue
                    if self._skip_test_on_param_should_not_be_none(param_with_none, func_name):
                        continue
                    if self._skip_test_on_overridden_class_param(clz_param_name, func_name):
                        continue
                    if self._skip_test_on_clz_param_excluded_in_func(clz_param_name, func_name):
                        continue
                    self._generate_and_verify_pipeline_definition(
                        target_func=getattr(obj, func_name),
                        expected_expr=expected_clz_expr,
                        param_with_none=param_with_none,
                    )

            # print("============================\n")
            self._default_args["clazz_args"][clz_param_name] = clz_default_arg

    def _iterate_func_params_to_check_compatibility(
        self,
        param_with_none: Optional[str] = None,
        test_func_for_none: Optional[str] = None,
    ):
        """Iterate each target func parameter and assign a pipeline var to it

        Args:
            param_with_none (str): The name of the parameter with None value.
            test_func_for_none (str): The name of the function which is being tested by
                replacing optional parameters to None.
        """
        obj = self._clazz(**self._default_args["clazz_args"])

        for func in self._target_funcs:
            func_name = func.__name__
            if test_func_for_none != func_name:
                # Iterating optional parameters of a specific target function,
                # which does not impact other target functions, so we can skip them
                continue
            if self._skip_test_on_param_should_not_be_none(param_with_none, func_name):
                continue
            print(
                f"#### Iterating parameters (supporting PipelineVariable) in target function: {func_name}"
            )
            func_params = {
                **self._func_params[func_name]["required"],
                **self._func_params[func_name]["optional"],
            }
            for func_param_name, func_default_arg in self._default_args["func_args"][
                func_name
            ].items():
                if func_param_name == param_with_none:
                    continue
                if self._skip_test_on_bonded_params(
                    param_name=func_param_name,
                    param_with_none=param_with_none,
                    target_func=func_name,
                ):
                    continue
                if not support_pipeline_variable(func_params[func_param_name]["type"]):
                    continue
                if self._skip_test_on_mutual_exclusive_params(func_param_name, func_name):
                    continue

                # For each arg which supports pipeline variables,
                # Replace it with each one of generated pipeline variables
                ppl_vars = generate_pipeline_vars_per_type(
                    func_param_name, func_params[func_param_name]["type"]
                )
                for func_ppl_var, expected_func_expr in ppl_vars:
                    print(
                        "Replacing func arg (%s) with pipeline variable which is expected to be (%s)"
                        % (func_param_name, expected_func_expr)
                    )
                    self._default_args["func_args"][func_name][func_param_name] = func_ppl_var
                    self._generate_and_verify_pipeline_definition(
                        target_func=getattr(obj, func_name),
                        expected_expr=expected_func_expr,
                        param_with_none=param_with_none,
                    )

                self._default_args["func_args"][func_name][func_param_name] = func_default_arg
                # print("-------------------------\n")

    def _generate_and_verify_pipeline_definition(
        self,
        target_func: callable,
        expected_expr: dict,
        param_with_none: str,
    ):
        """Generate a pipeline and verify the pipeline definition

        Args:
            target_func (callable): The function to generate step_args.
            expected_expr (dict): The expected json expression of a class or method argument.
            param_with_none (str): The name of the parameter with None value.
        """
        args = dict(
            name="MyStep",
            step_args=target_func(**self._default_args["func_args"][target_func.__name__]),
        )
        step = STEP_CLASS[self._clazz_type](**args)
        pipeline = Pipeline(
            name="MyPipeline",
            steps=[step],
            sagemaker_session=PIPELINE_SESSION,
        )
        step_dsl_obj = json.loads(pipeline.definition())["Steps"]
        step_dsl = json.dumps(step_dsl_obj)
        print("step_dsl:??", step_dsl)
        exp_origin = json.dumps(expected_expr["origin"])
        exp_to_str = json.dumps(expected_expr["to_string"])
        # if the testing arg is a dict, we may need to remove the outer {} of its expected expr
        # to compare, since for HyperParameters, some other arguments are auto inserted to the dict
        assert (
            exp_origin in step_dsl
            or exp_to_str in step_dsl
            or exp_origin[1:-1] in step_dsl
            or exp_to_str[1:-1] in step_dsl
        )

        # TODO: remove the following hard code assertion once recursive assignment is added
        if issubclass(self._clazz, Processor):
            if param_with_none != "network_config":
                assert json.dumps({"Get": "Parameters.nw_cfg_subnets"}) in step_dsl
                assert json.dumps({"Get": "Parameters.nw_cfg_security_group_ids"}) in step_dsl
                assert json.dumps({"Get": "Parameters.nw_cfg_enable_nw_isolation"}) in step_dsl
            if issubclass(self._clazz, SageMakerClarifyProcessor):
                if param_with_none != "data_config":
                    assert json.dumps({"Get": "Parameters.clarify_processor_input"}) in step_dsl
                    assert json.dumps({"Get": "Parameters.clarify_processor_output"}) in step_dsl
            else:
                if param_with_none != "outputs":
                    assert json.dumps({"Get": "Parameters.proc_output_source"}) in step_dsl
                    assert json.dumps({"Get": "Parameters.proc_output_dest"}) in step_dsl
                    assert json.dumps({"Get": "Parameters.proc_output_app_managed"}) in step_dsl
                if param_with_none != "inputs":
                    assert json.dumps({"Get": "Parameters.proc_input_source"}) in step_dsl
                    assert json.dumps({"Get": "Parameters.proc_input_dest"}) in step_dsl
                    assert json.dumps({"Get": "Parameters.proc_input_s3_data_type"}) in step_dsl
                    assert json.dumps({"Get": "Parameters.proc_input_app_managed"}) in step_dsl
        if issubclass(self._clazz, EstimatorBase):
            if isinstance(self._clazz, Estimator):
                if param_with_none != "debugger_hook_config":
                    assert json.dumps({"Get": "Parameters.debugger_hook_s3_output"}) in step_dsl
                if param_with_none != "profiler_config":
                    assert (
                            json.dumps({"Get": "Parameters.profile_config_system_monitor"}) in step_dsl
                    )
                if param_with_none != "tensorboard_output_config":
                    assert json.dumps({"Get": "Parameters.tensorboard_s3_output"}) in step_dsl
                if param_with_none != "inputs":
                    assert json.dumps({"Get": "Parameters.train_inputs_s3_data"}) in step_dsl
                    assert json.dumps({"Get": "Parameters.train_inputs_content_type"}) in step_dsl
                if param_with_none != "rules":
                    assert json.dumps({"Get": "Parameters.rules_instance_type"}) in step_dsl
            elif isinstance(self._clazz, AmazonAlgorithmEstimatorBase):
                if param_with_none != "records":
                    assert json.dumps({"Get": "Parameters.records_s3_data"}) in step_dsl
        if issubclass(self._clazz, HyperparameterTuner):
            if param_with_none != "inputs":
                assert json.dumps({"Get": "Parameters.inputs_estimator_1"}) in step_dsl
            if param_with_none != "warm_start_config":
                assert json.dumps({"Get": "Parameters.warm_start_cfg_parent"}) in step_dsl
            if param_with_none != "hyperparameter_ranges":
                assert (
                    json.dumps(
                        {
                            "Std:Join": {
                                "On": "",
                                "Values": [{"Get": "Parameters.hyper_range_min_value"}],
                            }
                        }
                    )
                    in step_dsl
                )
                assert (
                    json.dumps(
                        {
                            "Std:Join": {
                                "On": "",
                                "Values": [{"Get": "Parameters.hyper_range_max_value"}],
                            }
                        }
                    )
                    in step_dsl
                )
                assert json.dumps({"Get": "Parameters.hyper_range_scaling_type"}) in step_dsl
        if issubclass(self._clazz, Model):
            if step_dsl_obj[-1]["Type"] == "Model":
                if self._clazz.__name__ in {"Model", "FrameworkModel", "TensorFlowModel"}:
                    return
                if param_with_none != "serverless_inference_config":
                    assert json.dumps({"Get": "Parameters.serverless_cfg_memory_size"}) in step_dsl
                    assert (
                        json.dumps({"Get": "Parameters.serverless_cfg_max_concurrency"}) in step_dsl
                    )
            else:
                if param_with_none != "model_metrics":
                    assert (
                        json.dumps({"Get": "Parameters.model_statistics_content_type"}) in step_dsl
                    )
                    assert json.dumps({"Get": "Parameters.model_statistics_s3_uri"}) in step_dsl
                    assert (
                        json.dumps({"Get": "Parameters.model_statistics_content_digest"})
                        in step_dsl
                    )
                if param_with_none != "metadata_properties":
                    assert json.dumps({"Get": "Parameters.meta_properties_commit_id"}) in step_dsl
                    assert json.dumps({"Get": "Parameters.meta_properties_repository"}) in step_dsl
                    assert (
                        json.dumps({"Get": "Parameters.meta_properties_generated_by"}) in step_dsl
                    )
                    assert json.dumps({"Get": "Parameters.meta_properties_project_id"}) in step_dsl
                if param_with_none != "drift_check_baselines":
                    assert (
                        json.dumps({"Get": "Parameters.drift_constraints_content_type"}) in step_dsl
                    )
                    assert json.dumps({"Get": "Parameters.drift_constraints_s3_uri"}) in step_dsl
                    assert (
                        json.dumps({"Get": "Parameters.drift_constraints_content_digest"})
                        in step_dsl
                    )
                    assert json.dumps({"Get": "Parameters.drift_bias_content_type"}) in step_dsl
                    assert json.dumps({"Get": "Parameters.drift_bias_s3_uri"}) in step_dsl
                    assert json.dumps({"Get": "Parameters.drift_bias_content_digest"}) in step_dsl

    def _get_non_pipeline_val(self, n: str, t: type) -> object:
        """Get the value (not a Pipeline variable) based on parameter type and name

        Args:
            n (str): The parameter name. If a parameter has a pre-defined value,
                it will be returned directly.
            t (type): The parameter type. If a parameter does not have a pre-defined value,
                an arg will be auto-generated based on the type.

        Return:
            object: A Python primitive value is returned.
        """
        if n in FIXED_ARGUMENTS["common"]:
            return FIXED_ARGUMENTS["common"][n]
        if n in FIXED_ARGUMENTS[self._clazz_type]:
            return FIXED_ARGUMENTS[self._clazz_type][n]
        if t is str:
            return STR_VAL
        if t is int:
            return 1
        if t is float:
            return 1e-4
        if t is bool:
            return bool(getrandbits(1))
        if t in [list, tuple, dict, set]:
            return t()

        raise TypeError(f"Unable to parse type: {t}.")

    def _check_or_fill_in_args(self, params: dict, default_args: dict):
        """Check if every args are provided and not None

        Otherwise fill in with some default values

        Args:
            params (dict): The dict indicating the type of each parameter.
            default_args (dict): The dict of args to be checked or filled in.
        """
        for param_name, value in params.items():
            if param_name in default_args:
                # User specified the default value
                continue
            if value["default_value"]:
                # The parameter has default value in method definition
                default_args[param_name] = value["default_value"]
                continue
            clean_type = clean_up_types(value["type"])
            origin_type = get_origin(clean_type)
            if origin_type is None:
                default_args[param_name] = self._get_non_pipeline_val(param_name, clean_type)
            else:
                default_args[param_name] = self._get_non_pipeline_val(param_name, origin_type)

        self._check_or_update_default_args(default_args)

    def _check_or_update_default_args(self, default_args: dict):
        """To check if the default args are valid and update them if not

        Args:
            default_args (dict): The dict of args to be checked or updated.
        """
        if issubclass(self._clazz, EstimatorBase):
            if "disable_profiler" in default_args and default_args["disable_profiler"] is True:
                default_args["profiler_config"] = None

    def _get_target_functions(self) -> list:
        """Fetch the target functions based on class

        Return:
            list: The list of target functions is returned.
        """
        if issubclass(self._clazz, Processor):
            if issubclass(self._clazz, SageMakerClarifyProcessor):
                return [
                    self._clazz.run_pre_training_bias,
                    self._clazz.run_post_training_bias,
                    self._clazz.run_bias,
                    self._clazz.run_explainability,
                ]
            return [self._clazz.run]
        if issubclass(self._clazz, EstimatorBase):
            return [self._clazz.fit]
        if issubclass(self._clazz, Transformer):
            return [self._clazz.transform]
        if issubclass(self._clazz, HyperparameterTuner):
            return [self._clazz.fit]
        if issubclass(self._clazz, (Model, PipelineModel)):
            return [self._clazz.register, self._clazz.create]
        raise TypeError(f"Unable to get target function for class {self._clazz}")

    def _skip_test_on_bonded_params(
        self,
        param_name: str,
        param_with_none: str,
        target_func: str,
    ) -> bool:
        """Check if to skip testing with pipeline variables due to the bond relationship.

        I.e. the parameter (param_name) does not present in the definition json
        or it is not allowed to be a pipeline variable if its boned parameter is None.
        Then we can skip replacing the param_name with pipeline variables

        Args:
            param_name (str): The name of the parameter, which is to be verified that
                if we can skip replacing it with pipeline variables.
            param_with_none (str): The name of the parameter with None value.
            target_func (str): The target function impacted by the check.
                If target_func is init, it means the bonded parameters affect initiating
                the class object

        Return:
            bool: True if we can skip. False otherwise.
        """
        if self._clazz_type not in PARAM_BONDED_WITH_ANOTHER:
            return False
        if target_func not in PARAM_BONDED_WITH_ANOTHER[self._clazz_type]:
            return False
        for another_param in PARAM_BONDED_WITH_ANOTHER[self._clazz_type][target_func].get(
            param_name, {}
        ):
            if another_param == param_with_none:
                return True
            if target_func == "init":
                continue
            if self._default_args["clazz_args"].get(another_param, "NA") is None:
                return True
            if self._default_args["func_args"][target_func].get(another_param, "N/A") is None:
                return True
        return False

    def _skip_test_on_mutual_exclusive_params(self, param_name: str, target_func: str):
        """Check if to skip testing with pipeline variables due to mutual exclusive parameters.

        I.e. the parameter (param_name) should not be replaced with pipeline variables
        and tested on the target function (target_func), because it's mutual excluded
        by another not None parameter.

        Args:
            param_name (str): The name of the parameter, which is to be verified that
                if we can skip replacing it with pipeline variables.
            target_func (str): The target function impacted by the check.

        Return:
            bool: True if we can skip. False otherwise.
        """
        if self._clazz_type not in MUTUAL_EXCLUDED_PARAMS:
            return False
        if target_func not in MUTUAL_EXCLUDED_PARAMS[self._clazz_type]:
            return False
        for another_param in MUTUAL_EXCLUDED_PARAMS[self._clazz_type][target_func].get(
            param_name, {}
        ):
            if (
                another_param in self._default_args["clazz_args"]
                and self._default_args["clazz_args"][another_param] is not None
            ):
                return True
            if (
                another_param in self._default_args["func_args"][target_func]
                and self._default_args["func_args"][target_func][another_param] is not None
            ):
                return True
        return False

    def _skip_test_on_param_should_not_be_none(self, param_name: str, target_func: str):
        """Check if to skip testing due to the parameter should not be None.

        I.e. the parameter (param_name) is set to None in this round but it is not allowed
        according to the logic. Thus we can skip this round of test.

        Args:
            param_name (str): The name of the parameter, which is to be verified regarding None value.
            target_func (str): The target function impacted by this check.

        Return:
            bool: True if we can skip. False otherwise.
        """
        if self._clazz_type not in PARAMS_SHOULD_NOT_BE_NONE:
            return False
        if target_func not in PARAMS_SHOULD_NOT_BE_NONE[self._clazz_type]:
            return False
        return param_name in PARAMS_SHOULD_NOT_BE_NONE[self._clazz_type][target_func]

    def _skip_test_on_overridden_class_param(self, clz_param_name: str, target_func: str):
        """Check if to skip testing with pipeline variables on class parameter due to override.

        I.e. the class parameter (clz_param_name) should not be replaced with pipeline variables
        and tested on the target function (target_func) because it's overridden by a
        function parameter with the same name.
        e.g. image_uri in model.create can override that in model constructor.

        Args:
            clz_param_name (str): The name of the class parameter, which is to be verified.
            target_func (str): The target function impacted by the check.

        Return:
            bool: True if we can skip. False otherwise.
        """
        return self._default_args["func_args"][target_func].get(clz_param_name, None) is not None

    def _skip_test_on_clz_param_excluded_in_func(self, clz_param_name: str, target_func: str):
        """Check if to skip testing with pipeline variables on class parameter due to exclusion.

        I.e. the class parameter (clz_param_name) should not be replaced with pipeline variables
        and tested on the target function (target_func), as it's not used there.

        Args:
            clz_param_name (str): The name of the class parameter, which is to be verified.
            target_func (str): The target function impacted by the check.

        Return:
            bool: True if we can skip. False otherwise.
        """
        if self._clazz_type not in CLASS_PARAMS_EXCLUDED_IN_TARGET_FUNC:
            return False
        if target_func not in CLASS_PARAMS_EXCLUDED_IN_TARGET_FUNC[self._clazz_type]:
            return False
        return clz_param_name in CLASS_PARAMS_EXCLUDED_IN_TARGET_FUNC[self._clazz_type][target_func]

    def _skip_test_on_base_clz_param_excluded_in_subclz(self, clz_param_name: str):
        """Check if to skip testing with pipeline variables on class parameter due to exclusion.

        I.e. the base class parameter (clz_param_name) should not be replaced with pipeline variables,
        as it's not used in the subclass.

        Args:
            clz_param_name (str): The name of the class parameter, which is to be verified.

        Return:
            bool: True if we can skip. False otherwise.
        """
        if self._clazz_type not in BASE_CLASS_PARAMS_EXCLUDED_IN_SUB_CLASS:
            return False
        if self._clazz.__name__ not in BASE_CLASS_PARAMS_EXCLUDED_IN_SUB_CLASS[self._clazz_type]:
            return False
        return (
            clz_param_name
            in BASE_CLASS_PARAMS_EXCLUDED_IN_SUB_CLASS[self._clazz_type][self._clazz.__name__]
        )
