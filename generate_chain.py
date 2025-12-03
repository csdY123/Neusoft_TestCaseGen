from langchain_core.language_models import BaseLanguageModel
from openai import OpenAI

import gradio as gr

from model_util import generate_response, generate_response_vllm, generate_response_vllm_stream
from parse_util import robust_json_parse
from prompt_util import load_prompt_template


# >>>>>>>> Feature Generation Start <<<<<<<<
def generate_features_for_gradio(global_data: dict, llm, prd_text, additional_requirement="",
                                 system_prompt_name="generate_features_system_prompt",
                                 user_prompt_name="generate_features_user_prompt",
                                 use_vllm=False, vllm_client=None, model_id=None):
    """Generate features and update Gradio interface (non-streaming)"""
    output, thinking, features, features_choices = generate_features(
        llm, prd_text, additional_requirement, system_prompt_name, user_prompt_name,
        use_vllm=use_vllm, vllm_client=vllm_client, model_id=model_id
    )

    global_data["prd_text"] = prd_text
    global_data["features"] = features

    return output, thinking, gr.Dropdown(choices=features_choices), gr.Dropdown(choices=features_choices)


def generate_features_for_gradio_stream(global_data: dict, vllm_client: OpenAI, model_id: str,
                                        prd_text, additional_requirement="",
                                        system_prompt_name="generate_features_system_prompt",
                                        user_prompt_name="generate_features_user_prompt"):
    """Generate features with streaming output for Gradio"""
    if not prd_text or not prd_text.strip():
        yield "⚠️ Please input PRD document", "", gr.Dropdown(choices=[]), gr.Dropdown(choices=[])
        return

    system_prompt = load_prompt_template(system_prompt_name)
    user_prompt_template = load_prompt_template(user_prompt_name)
    user_prompt = user_prompt_template.format(prd_text=prd_text)

    if additional_requirement:
        user_prompt += f"\n\nAdditional requirement: {additional_requirement}"

    # Streaming generation
    response_text = ""
    for partial_response in generate_response_vllm_stream(vllm_client, model_id, user_prompt, system_prompt):
        response_text = partial_response
        yield f"🔄 Generating...\n\n**Raw output:**\n```\n{partial_response}\n```", "", gr.Dropdown(choices=[]), gr.Dropdown(choices=[])

    # Parse after generation complete
    output, thinking, features, features_choices = parse_features(response_text)

    global_data["prd_text"] = prd_text
    global_data["features"] = features

    yield output, thinking, gr.Dropdown(choices=features_choices), gr.Dropdown(choices=features_choices)


def generate_features(llm, prd_text, additional_requirement="",
                      system_prompt_name="generate_features_system_prompt",
                      user_prompt_name="generate_features_user_prompt",
                      use_vllm=False, vllm_client=None, model_id=None):
    """Generate features from PRD document"""

    system_prompt = load_prompt_template(system_prompt_name)
    user_prompt_template = load_prompt_template(user_prompt_name)
    user_prompt = user_prompt_template.format(prd_text=prd_text)

    if additional_requirement:
        user_prompt += f"\n\nAdditional requirement: {additional_requirement}"

    # Generate features
    if use_vllm and vllm_client and model_id:
        response = generate_response_vllm(vllm_client, model_id, user_prompt, system_prompt)
    else:
        response = generate_response(llm, user_prompt, system_prompt, enable_thinking=False)

    return parse_features(response)


def parse_features(response, prd_text=""):
    """Parse features from response"""
    try:
        result = robust_json_parse(response)

        if not isinstance(result, dict) or "features" not in result:
            raise ValueError("Parse error: missing 'features' field")

        features = result.get("features", [])

        if features is None or len(features) == 0:
            raise ValueError("No features generated")

        output = "## Generated Features\n\n"
        for feature in features:
            output += f"### {feature.get('id', '?')}. {feature.get('name', 'Unknown')}\n"
            output += f"{feature.get('description', 'No description')}\n\n"

        features_choices = [f"{feature['id']}. {feature['name']}" for feature in features]

        return output, "✅ Generation complete", features, features_choices
    except Exception as e:
        error_msg = f"⚠️ Parse failed\n\n**Error:** {str(e)}\n\n**Raw output:**\n```\n{response[:500]}{'...' if len(str(response)) > 500 else ''}\n```"
        return error_msg, f"Error: {str(e)}", [], []


# >>>>>>>> Feature Generation End <<<<<<<<

# >>>>>>>> Test Point Generation Start <<<<<<<<

def get_test_point_choices(global_data: dict, feature_choice):
    """Get test point choices list"""
    if not feature_choice:
        return []

    feature_id = int(feature_choice.split(".")[0])
    feature_idx = feature_id - 1

    if feature_idx not in global_data["test_points"]:
        return []

    return [f"{tp['id']}. {tp['name']}" for tp in global_data["test_points"][feature_idx]]


def generate_test_points_for_gradio(global_data: dict, llm, feature_choice, additional_requirement,
                                    current_feature_choice_step3,
                                    use_vllm=False, vllm_client=None, model_id=None):
    print("start generate_test_points_for_gradio ...")

    prd_text = global_data.get("prd_text", "")
    features = global_data.get("features", [])

    feature_id = int(feature_choice.split(".")[0])
    feature_idx = feature_id - 1
    if feature_idx < 0 or feature_idx >= len(features):
        return "⚠️ Invalid feature index", ""

    feature = features[feature_idx]

    output, thinking, test_points = generate_test_points(
        llm, prd_text, feature, additional_requirement,
        use_vllm=use_vllm, vllm_client=vllm_client, model_id=model_id
    )

    global_data["test_points"][feature_idx] = test_points

    if current_feature_choice_step3 and current_feature_choice_step3 == feature_choice:
        test_point_choices = get_test_point_choices(global_data, feature_choice)
        return output, thinking, gr.Dropdown(choices=test_point_choices)
    else:
        return output, thinking, gr.Dropdown()


def generate_test_points_for_gradio_stream(global_data: dict, vllm_client: OpenAI, model_id: str,
                                           feature_choice, additional_requirement,
                                           current_feature_choice_step3,
                                           system_prompt_name="generate_test_points_system_prompt",
                                           user_prompt_name="generate_test_points_user_prompt"):
    """Generate test points with streaming output for Gradio"""
    if not feature_choice:
        yield "⚠️ Please select a feature", "", gr.Dropdown()
        return

    prd_text = global_data.get("prd_text", "")
    features = global_data.get("features", [])

    feature_id = int(feature_choice.split(".")[0])
    feature_idx = feature_id - 1
    if feature_idx < 0 or feature_idx >= len(features):
        yield "⚠️ Invalid feature index", "", gr.Dropdown()
        return

    feature = features[feature_idx]

    system_prompt = load_prompt_template(system_prompt_name)
    user_prompt_template = load_prompt_template(user_prompt_name)
    user_prompt = user_prompt_template.format(
        feature_name=feature["name"],
        feature_description=feature["description"],
        prd_text=prd_text
    )

    if additional_requirement:
        user_prompt += f"\n\nAdditional requirement: {additional_requirement}"

    # Streaming generation
    response_text = ""
    for partial_response in generate_response_vllm_stream(vllm_client, model_id, user_prompt, system_prompt):
        response_text = partial_response
        yield f"🔄 Generating...\n\n**Raw output:**\n```\n{partial_response}\n```", "", gr.Dropdown()

    # Parse after generation complete
    output, thinking, test_points = parse_test_points(feature, response_text)

    global_data["test_points"][feature_idx] = test_points

    if current_feature_choice_step3 and current_feature_choice_step3 == feature_choice:
        test_point_choices = get_test_point_choices(global_data, feature_choice)
        yield output, thinking, gr.Dropdown(choices=test_point_choices)
    else:
        yield output, thinking, gr.Dropdown()


def generate_test_points(llm, prd_text, feature, additional_requirement="",
                         system_prompt_name="generate_test_points_system_prompt",
                         user_prompt_name="generate_test_points_user_prompt",
                         use_vllm=False, vllm_client=None, model_id=None):
    system_prompt = load_prompt_template(system_prompt_name)
    user_prompt_template = load_prompt_template(user_prompt_name)
    user_prompt = user_prompt_template.format(feature_name=feature["name"], feature_description=feature["description"],
                                              prd_text=prd_text)

    if additional_requirement:
        user_prompt += f"\n\nAdditional requirement: {additional_requirement}"

    # Generate test points
    if use_vllm and vllm_client and model_id:
        response = generate_response_vllm(vllm_client, model_id, user_prompt, system_prompt)
    else:
        response = generate_response(llm, user_prompt, system_prompt, enable_thinking=False)

    return parse_test_points(feature, response)


def parse_test_points(feature, response):
    """Parse test points from response"""
    try:
        result = robust_json_parse(response)

        if not isinstance(result, dict) or "test_points" not in result:
            raise ValueError("Parse error: missing 'test_points' field")

        test_points = result.get("test_points", [])

        if not test_points:
            raise ValueError("No test points generated")

        output = f"## Feature: {feature['name']}\n\n"
        output += "### Test Points\n\n"

        for test_point in test_points:
            output += f"#### {test_point.get('id', '?')}. {test_point.get('name', 'Unknown')}\n"
            output += f"- **Type**: {test_point.get('type', 'Unspecified')}\n"
            output += f"- **Priority**: {test_point.get('priority', 'Unspecified')}\n"
            output += f"- **Description**: {test_point.get('description', 'No description')}\n"
            output += f"- **Precondition**: {test_point.get('precondition', 'None')}\n"
            output += f"- **Expected Result**: {test_point.get('expected_result', 'None')}\n\n"

        return output, "✅ Generation complete", test_points

    except Exception as e:
        error_msg = f"⚠️ Parse failed\n\n**Error:** {str(e)}\n\n**Raw output:**\n```\n{response[:500]}{'...' if len(str(response)) > 500 else ''}\n```"
        return error_msg, f"Error: {str(e)}", []


# >>>>>>>> Test Point Generation End <<<<<<<<

# >>>>>>>> Test Case Generation Start <<<<<<<<
def generate_test_cases_for_gradio(global_data: dict, llm,
                                   feature_choice, tp_choice, additional_requirement,
                                   use_vllm=False, vllm_client=None, model_id=None):
    """Generate test cases for selected test point"""

    if not feature_choice or not tp_choice:
        return "⚠️ Please select feature and test point", ""
    feature_index = int(feature_choice.split(".")[0])
    test_point_index = int(tp_choice.split(".")[0])

    feature_idx = int(feature_index) - 1
    test_point_idx = int(test_point_index) - 1

    if feature_idx not in global_data["test_points"]:
        return "⚠️ Test points not generated for this feature", ""

    test_points = global_data["test_points"][feature_idx]
    if test_point_idx < 0 or test_point_idx >= len(test_points):
        return "⚠️ Invalid test point index", ""

    feature = global_data["features"][feature_idx]
    test_point = test_points[test_point_idx]

    output, thinking, test_cases = generate_test_cases(
        llm, global_data['prd_text'], feature, test_point, additional_requirement,
        use_vllm=use_vllm, vllm_client=vllm_client, model_id=model_id
    )

    global_data["test_cases"][(feature_idx, test_point_idx)] = test_cases

    return output, thinking


def generate_test_cases_for_gradio_stream(global_data: dict, vllm_client: OpenAI, model_id: str,
                                          feature_choice, tp_choice, additional_requirement,
                                          system_prompt_name="generate_test_cases_system_prompt",
                                          user_prompt_name="generate_test_cases_user_prompt"):
    """Generate test cases with streaming output for Gradio"""
    if not feature_choice or not tp_choice:
        yield "⚠️ Please select feature and test point", ""
        return

    feature_index = int(feature_choice.split(".")[0])
    test_point_index = int(tp_choice.split(".")[0])

    feature_idx = int(feature_index) - 1
    test_point_idx = int(test_point_index) - 1

    if feature_idx not in global_data["test_points"]:
        yield "⚠️ Test points not generated for this feature", ""
        return

    test_points = global_data["test_points"][feature_idx]
    if test_point_idx < 0 or test_point_idx >= len(test_points):
        yield "⚠️ Invalid test point index", ""
        return

    feature = global_data["features"][feature_idx]
    test_point = test_points[test_point_idx]

    system_prompt = load_prompt_template(system_prompt_name)
    user_prompt_template = load_prompt_template(user_prompt_name)
    user_prompt = user_prompt_template.format(
        feature_name=feature["name"],
        test_point_name=test_point["name"],
        test_point_description=test_point["description"],
        test_point_type=test_point["type"],
        test_point_priority=test_point["priority"],
        test_point_precondition=test_point["precondition"],
        test_point_expected_result=test_point["expected_result"],
        prd_text=global_data['prd_text']
    )

    if additional_requirement:
        user_prompt += f"\n\nAdditional requirement: {additional_requirement}"

    # Streaming generation
    response_text = ""
    for partial_response in generate_response_vllm_stream(vllm_client, model_id, user_prompt, system_prompt):
        response_text = partial_response
        yield f"🔄 Generating...\n\n**Raw output:**\n```\n{partial_response}\n```", ""

    # Parse after generation complete
    output, thinking, test_cases = parse_test_cases(response_text, feature, test_point)

    global_data["test_cases"][(feature_idx, test_point_idx)] = test_cases

    yield output, thinking


def generate_test_cases(llm, prd_text, feature, test_point, additional_requirement="",
                        system_prompt_name="generate_test_cases_system_prompt",
                        user_prompt_name="generate_test_cases_user_prompt",
                        use_vllm=False, vllm_client=None, model_id=None):
    system_prompt = load_prompt_template(system_prompt_name)
    user_prompt_template = load_prompt_template(user_prompt_name)
    user_prompt = user_prompt_template.format(feature_name=feature["name"], test_point_name=test_point["name"],
                                              test_point_description=test_point["description"],
                                              test_point_type=test_point["type"],
                                              test_point_priority=test_point["priority"],
                                              test_point_precondition=test_point["precondition"],
                                              test_point_expected_result=test_point["expected_result"],
                                              prd_text=prd_text)

    if additional_requirement:
        user_prompt += f"\n\nAdditional requirement: {additional_requirement}"

    # Generate test cases
    if use_vllm and vllm_client and model_id:
        response = generate_response_vllm(vllm_client, model_id, user_prompt, system_prompt)
    else:
        response = generate_response(llm, user_prompt, system_prompt, enable_thinking=False)

    return parse_test_cases(response, feature, test_point)


def parse_test_cases(response, feature, test_point):
    """Parse test cases from response"""
    try:
        result = robust_json_parse(response)

        if not isinstance(result, dict) or "test_cases" not in result:
            raise ValueError("Parse error: missing 'test_cases' field")

        test_cases = result.get("test_cases", [])

        if not test_cases:
            raise ValueError("No test cases generated")

        output = f"## Feature: {feature['name']}\n"
        output += f"### Test Point: {test_point['name']}\n\n"
        output += "### Test Cases\n\n"

        for tc in test_cases:
            output += f"#### {tc.get('case_id', '?')}: {tc.get('title', 'Unknown')}\n"
            output += f"- **Priority**: {tc.get('priority', 'Unspecified')}\n"
            output += f"- **Precondition**: {tc.get('precondition', 'None')}\n"
            output += f"- **Test Steps**:\n"

            test_steps = tc.get('test_steps', [])
            if test_steps:
                for step in test_steps:
                    output += f"  {step.get('step', '?')}. {step.get('action', 'No action')}\n"
                    output += f"     - Expected: {step.get('expected', 'None')}\n"
            else:
                output += f"  (No test steps)\n"

            output += f"- **Test Data**: {tc.get('test_data', 'None')}\n"
            output += f"- **Expected Result**: {tc.get('expected_result', 'None')}\n"
            output += f"- **Postcondition**: {tc.get('postcondition', 'None')}\n\n"

        return output, "✅ Generation complete", test_cases

    except Exception as e:
        error_msg = f"⚠️ Parse failed\n\n**Error:** {str(e)}\n\n**Raw output:**\n```\n{response[:500]}{'...' if len(str(response)) > 500 else ''}\n```"
        return error_msg, f"Error: {str(e)}", []

# >>>>>>>> Test Case Generation End <<<<<<<<
