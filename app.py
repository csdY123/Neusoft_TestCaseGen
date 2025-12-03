import json
from functools import partial
import os
import tempfile
import gradio as gr
from langchain_ollama import OllamaLLM

from model_util import create_vllm_client
from doc_util import (
    get_document_choices, get_document_by_display_name, get_document_content,
    save_uploaded_document, uploaded_documents
)
from generate_chain import (
    generate_features_for_gradio, generate_features_for_gradio_stream,
    generate_test_points_for_gradio, generate_test_points_for_gradio_stream,
    generate_test_cases_for_gradio, generate_test_cases_for_gradio_stream,
    get_test_point_choices
)
from export_util import (
    export_to_json_string, save_to_server, get_saved_exports,
    format_data_for_visualization, format_data_for_labeling, load_export_file
)
from rating_util import save_rating, get_ratings, get_rating_summary, export_ratings_csv

# Model configuration
MODEL_CONFIG = {
    "ollama": {
        "model": "qwen3:8b",
        "llm": None
    },
    "vllm": {
        "base_url": "http://localhost:12349/v1",
        "model_id": "Qwen3-8B",
        "client": None
    }
}

# Global data
global_data = {
    "prd_text": "",
    "features": [],
    "test_points": {},  # {feature_index: [test_points]}
    "test_cases": {},  # {(feature_index, test_point_index): [test_cases]}
    "document_id": None,
    "document_display_name": ""
}


def init_ollama():
    """Initialize Ollama model"""
    MODEL_CONFIG["ollama"]["llm"] = OllamaLLM(model=MODEL_CONFIG["ollama"]["model"])
    return "✅ Ollama model initialized"


def init_vllm(base_url, model_id):
    """Initialize vLLM client"""
    MODEL_CONFIG["vllm"]["base_url"] = base_url
    MODEL_CONFIG["vllm"]["model_id"] = model_id
    MODEL_CONFIG["vllm"]["client"] = create_vllm_client(base_url)
    return f"✅ vLLM client initialized (URL: {base_url}, Model: {model_id})"


def get_feature_choices_list(global_data):
    """Get feature choices list"""
    if not global_data["features"]:
        return []
    return [f"{f['id']}. {f['name']}" for f in global_data["features"]]


def export_all_data(global_data):
    """Export all generated data as JSON"""
    export_data = {
        "document": {
            "id": global_data.get("document_id"),
            "display_name": global_data.get("document_display_name")
        },
        "prd_text": global_data["prd_text"],
        "features": global_data["features"],
        "test_points": global_data["test_points"],
        "test_cases": {}
    }

    for key, value in global_data["test_cases"].items():
        if isinstance(key, tuple):
            str_key = f"{key[0]},{key[1]}"
            export_data["test_cases"][str_key] = value
        else:
            export_data["test_cases"][key] = value

    return json.dumps(export_data, ensure_ascii=False, indent=2)


# Document upload handlers
def handle_file_upload(files):
    """Handle file upload and update dropdowns"""
    if not files:
        choices = get_document_choices()
        value = choices[-1] if choices else None
        preview = get_document_content(value) if value else ""
        return gr.Dropdown(choices=choices, value=value), preview, gr.Dropdown(choices=choices, value=value), preview

    if not isinstance(files, list):
        files = [files]

    last_choice = None
    for file_data in files:
        if file_data is None:
            continue
        try:
            doc_id = save_uploaded_document(file_data)
            doc = uploaded_documents[doc_id]
            last_choice = doc["display_name"]
        except Exception:
            pass

    choices = get_document_choices()
    value = last_choice if last_choice else (choices[-1] if choices else None)
    preview = get_document_content(value) if value else ""
    return gr.Dropdown(choices=choices, value=value), preview, gr.Dropdown(choices=choices, value=value), preview


def on_doc_dropdown_change(choice):
    """Handle document dropdown change"""
    preview = get_document_content(choice) if choice else ""
    choices = get_document_choices()
    return preview, gr.Dropdown(choices=choices, value=choice), preview


def on_prd_doc_change(choice):
    """Handle PRD document dropdown change"""
    return get_document_content(choice) if choice else ""


def init_gradio_page():
    # Get initial document choices
    initial_choices = get_document_choices()
    initial_value = initial_choices[-1] if initial_choices else None
    initial_preview = get_document_content(initial_value) if initial_value else ""

    with gr.Blocks(title="PRD to Test Case Generation System") as demo:
        gr.Markdown("""
        # 🧪 PRD to Test Case Generation System
        
        Automated test case generation tool supporting end-to-end generation from PRD to features, test points, and test cases.
        
        ## Usage:
        1. **Configure Model**: Select model backend (Ollama or vLLM) and initialize
        2. **Upload Document**: Upload PRD document (.docx) or paste text directly
        3. **Generate Features**: Click generate, regenerate if not satisfied
        4. **Generate Test Points**: Select feature, generate corresponding test points
        5. **Generate Test Cases**: Select test point, generate detailed test cases
        6. **Export Data**: Export all generated structured data
        """)

        gr.Markdown("---")

        # Model Configuration Section
        with gr.Accordion("⚙️ Model Configuration", open=True):
            with gr.Row():
                model_backend = gr.Radio(
                    label="Model Backend",
                    choices=["vLLM (Streaming)", "Ollama"],
                    value="vLLM (Streaming)"
                )
            with gr.Row():
                with gr.Column(visible=True) as vllm_config:
                    vllm_url = gr.Textbox(
                        label="vLLM API URL",
                        value="http://localhost:12349/v1",
                        placeholder="http://localhost:8000/v1"
                    )
                    vllm_model = gr.Textbox(
                        label="Model ID",
                        value="Qwen3-8B",
                        placeholder="Qwen3-8B"
                    )
                    init_vllm_btn = gr.Button("🚀 Initialize vLLM", variant="primary")
                with gr.Column(visible=False) as ollama_config:
                    ollama_model = gr.Textbox(
                        label="Ollama Model",
                        value="qwen3:8b",
                        placeholder="qwen3:8b"
                    )
                    init_ollama_btn = gr.Button("🚀 Initialize Ollama", variant="primary")

            model_status = gr.Textbox(label="Model Status", value="⏳ Model not initialized", interactive=False)

            def toggle_model_config(backend):
                if backend == "vLLM (Streaming)":
                    return gr.Column(visible=True), gr.Column(visible=False)
                else:
                    return gr.Column(visible=False), gr.Column(visible=True)

            model_backend.change(
                fn=toggle_model_config,
                inputs=model_backend,
                outputs=[vllm_config, ollama_config]
            )

            init_vllm_btn.click(fn=init_vllm, inputs=[vllm_url, vllm_model], outputs=model_status)

            def init_ollama_with_model(model_name):
                MODEL_CONFIG["ollama"]["model"] = model_name
                return init_ollama()

            init_ollama_btn.click(fn=init_ollama_with_model, inputs=ollama_model, outputs=model_status)

        gr.Markdown("---")

        # Step 0: Upload Document
        with gr.Tab("📁 Step 0: Upload Document"):
            with gr.Row():
                doc_upload = gr.File(
                    label="Upload PRD Document (.docx, multiple allowed)",
                    file_types=[".docx"],
                    file_count="multiple"
                )
            with gr.Row():
                with gr.Column(scale=1):
                    uploaded_doc_dropdown = gr.Dropdown(
                        label="Uploaded Documents",
                        choices=initial_choices,
                        value=initial_value,
                        interactive=True
                    )
                with gr.Column(scale=2):
                    doc_preview = gr.Textbox(
                        label="Document Content",
                        value=initial_preview,
                        lines=30,
                        max_lines=1000,
                        interactive=False,
                        show_copy_button=True
                    )

        # Step 1: PRD -> Features
        with gr.Tab("📄 Step 1: PRD → Features"):
            with gr.Row():
                with gr.Column(scale=2):
                    prd_doc_dropdown = gr.Dropdown(
                        label="Select PRD Document",
                        choices=initial_choices,
                        value=initial_value,
                        interactive=True
                    )
                    prd_doc_preview = gr.Textbox(
                        label="Document Content Preview",
                        value=initial_preview,
                        lines=12,
                        max_lines=1000,
                        interactive=False,
                        show_copy_button=True
                    )
                    feature_requirement = gr.Textbox(
                        label="Additional Requirements (Optional)",
                        placeholder="E.g., Focus on user interaction features...",
                        lines=2
                    )
                    with gr.Row():
                        gen_feature_btn = gr.Button("✨ Generate Features", variant="primary")
                        re_gen_feature_btn = gr.Button("🔄 Regenerate Features")

                with gr.Column(scale=3):
                    feature_output = gr.Markdown(label="Generated Features")
                    feature_thinking = gr.Textbox(label="Model Status", lines=2)
                    
                    # Manual edit section
                    with gr.Accordion("✏️ Edit Features", open=False):
                        feature_edit_json = gr.Textbox(
                            label="Edit Features (JSON)",
                            placeholder='[{"id": 1, "name": "Feature Name", "description": "Description"}]',
                            lines=10
                        )
                        with gr.Row():
                            load_features_btn = gr.Button("📥 Load Current")
                            save_features_btn = gr.Button("💾 Save Changes", variant="primary")
                        feature_edit_status = gr.Textbox(label="", interactive=False, lines=1)
                    
                    # Manual rating section
                    with gr.Accordion("⭐ Rate Features", open=False):
                        feature_rating = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3,
                            label="Quality Score (1-5)"
                        )
                        feature_comment = gr.Textbox(
                            label="Comments",
                            placeholder="Add your feedback...",
                            lines=2
                        )
                        save_feature_rating_btn = gr.Button("💾 Save Rating")
                        feature_rating_status = gr.Textbox(label="", interactive=False, lines=1)

        # Document upload event bindings
        doc_upload.upload(
            fn=handle_file_upload,
            inputs=doc_upload,
            outputs=[uploaded_doc_dropdown, doc_preview, prd_doc_dropdown, prd_doc_preview]
        )

        uploaded_doc_dropdown.change(
            fn=on_doc_dropdown_change,
            inputs=uploaded_doc_dropdown,
            outputs=[doc_preview, prd_doc_dropdown, prd_doc_preview]
        )

        prd_doc_dropdown.change(
            fn=on_prd_doc_change,
            inputs=prd_doc_dropdown,
            outputs=prd_doc_preview
        )

        # Step 2: Features -> Test Points
        with gr.Tab("🎯 Step 2: Features → Test Points"):
            with gr.Row():
                with gr.Column(scale=1):
                    feature_dropdown = gr.Dropdown(
                        label="Select Feature",
                        choices=[],
                        interactive=True
                    )
                    test_point_requirement = gr.Textbox(
                        label="Additional Requirements (Optional)",
                        placeholder="E.g., Add performance test points...",
                        lines=2
                    )
                    with gr.Row():
                        gen_tp_btn = gr.Button("✨ Generate Test Points", variant="primary")
                        regen_tp_btn = gr.Button("🔄 Regenerate Test Points")

                    refresh_feature_btn = gr.Button("🔄 Refresh Feature List")

                with gr.Column(scale=3):
                    test_point_output = gr.Markdown(label="Generated Test Points")
                    test_point_thinking = gr.Textbox(label="Model Status", lines=2)
                    
                    # Manual edit section
                    with gr.Accordion("✏️ Edit Test Points", open=False):
                        tp_edit_json = gr.Textbox(
                            label="Edit Test Points (JSON)",
                            placeholder='[{"id": 1, "name": "Test Point", "type": "Functional", "priority": "High", "description": "...", "precondition": "...", "expected_result": "..."}]',
                            lines=10
                        )
                        with gr.Row():
                            load_tp_btn = gr.Button("📥 Load Current")
                            save_tp_btn = gr.Button("💾 Save Changes", variant="primary")
                        tp_edit_status = gr.Textbox(label="", interactive=False, lines=1)
                    
                    # Manual rating section
                    with gr.Accordion("⭐ Rate Test Points", open=False):
                        tp_rating = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3,
                            label="Quality Score (1-5)"
                        )
                        tp_comment = gr.Textbox(
                            label="Comments",
                            placeholder="Add your feedback...",
                            lines=2
                        )
                        save_tp_rating_btn = gr.Button("💾 Save Rating")
                        tp_rating_status = gr.Textbox(label="", interactive=False, lines=1)

            def update_feature_dropdown(global_data):
                choices = get_feature_choices_list(global_data)
                return gr.Dropdown(choices=choices)

            refresh_feature_btn.click(
                fn=partial(update_feature_dropdown, global_data),
                outputs=feature_dropdown
            )

        # Step 3: Test Points -> Test Cases
        with gr.Tab("📝 Step 3: Test Points → Test Cases"):
            with gr.Row():
                with gr.Column(scale=1):
                    feature_dropdown2 = gr.Dropdown(
                        label="Select Feature",
                        choices=[],
                        interactive=True
                    )
                    test_point_dropdown = gr.Dropdown(
                        label="Select Test Point",
                        choices=[],
                        interactive=True
                    )
                    test_case_requirement = gr.Textbox(
                        label="Additional Requirements (Optional)",
                        placeholder="E.g., Add exception scenario test cases...",
                        lines=2
                    )
                    with gr.Row():
                        gen_tc_btn = gr.Button("✨ Generate Test Cases", variant="primary")
                        regen_tc_btn = gr.Button("🔄 Regenerate Test Cases")

                    with gr.Row():
                        refresh_feature_btn2 = gr.Button("🔄 Refresh Feature List")
                        refresh_test_point_btn = gr.Button("🔄 Refresh Test Point List")

                with gr.Column(scale=3):
                    test_case_output = gr.Markdown(label="Generated Test Cases")
                    test_case_thinking = gr.Textbox(label="Model Status", lines=2)
                    
                    # Manual edit section
                    with gr.Accordion("✏️ Edit Test Cases", open=False):
                        tc_edit_json = gr.Textbox(
                            label="Edit Test Cases (JSON)",
                            placeholder='[{"case_id": "TC001", "title": "...", "priority": "High", "precondition": "...", "test_steps": [...], "test_data": "...", "expected_result": "...", "postcondition": "..."}]',
                            lines=12
                        )
                        with gr.Row():
                            load_tc_btn = gr.Button("📥 Load Current")
                            save_tc_btn = gr.Button("💾 Save Changes", variant="primary")
                        tc_edit_status = gr.Textbox(label="", interactive=False, lines=1)
                    
                    # Manual rating section
                    with gr.Accordion("⭐ Rate Test Cases", open=False):
                        tc_rating = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3,
                            label="Quality Score (1-5)"
                        )
                        tc_comment = gr.Textbox(
                            label="Comments",
                            placeholder="Add your feedback...",
                            lines=2
                        )
                        save_tc_rating_btn = gr.Button("💾 Save Rating")
                        tc_rating_status = gr.Textbox(label="", interactive=False, lines=1)

            refresh_feature_btn2.click(
                fn=partial(update_feature_dropdown, global_data),
                outputs=feature_dropdown2
            )

            def update_test_point_dropdown(global_data, feature_choice):
                if not feature_choice:
                    return gr.Dropdown(choices=[])
                choices = get_test_point_choices(global_data, feature_choice)
                return gr.Dropdown(choices=choices)

            feature_dropdown2.change(
                fn=partial(update_test_point_dropdown, global_data),
                inputs=feature_dropdown2,
                outputs=test_point_dropdown
            )

            refresh_test_point_btn.click(
                fn=partial(update_test_point_dropdown, global_data),
                inputs=feature_dropdown2,
                outputs=test_point_dropdown
            )

        # Step 4: Data Export
        with gr.Tab("💾 Data Export"):
            gr.Markdown("""
            ### Export All Generated Data
            Export format is JSON, containing PRD text, features, test points and test cases.
            Can be used for data annotation, model training or import to test management system.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 📥 Export Actions")
                    with gr.Row():
                        export_json_btn = gr.Button("📄 Export JSON", variant="primary")
                        save_server_btn = gr.Button("💾 Save to Server", variant="secondary")
                    
                    download_file = gr.File(label="Download JSON File", interactive=False)
                    save_status = gr.Textbox(label="Save Status", interactive=False, lines=2)
                    
                    gr.Markdown("#### 📂 Saved Exports")
                    saved_exports_dropdown = gr.Dropdown(
                        label="Select Saved Export",
                        choices=[],
                        interactive=True
                    )
                    refresh_exports_btn = gr.Button("🔄 Refresh List")
                    load_export_btn = gr.Button("📂 Load Selected Export")

                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("📊 Visualization"):
                            visualization_output = gr.Markdown(label="Data Visualization")
                        with gr.Tab("🏷️ Labeling View"):
                            labeling_output = gr.Markdown(label="Labeling View")
                        with gr.Tab("📝 Raw JSON"):
                            export_output = gr.Textbox(
                                label="Exported JSON Data",
                                lines=25,
                                max_lines=500,
                                show_copy_button=True
                            )

            # Export handlers
            def export_json_handler(global_data):
                json_str = export_to_json_string(global_data)
                vis = format_data_for_visualization(global_data)
                label = format_data_for_labeling(global_data)
                
                # Create temp file for download
                temp_file = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8"
                )
                temp_file.write(json_str)
                temp_file.close()
                
                return temp_file.name, "", vis, label, json_str

            def save_to_server_handler(global_data):
                try:
                    filepath = save_to_server(global_data)
                    return f"✅ Saved to: {filepath}"
                except Exception as e:
                    return f"❌ Error: {str(e)}"

            def refresh_exports_handler():
                exports = get_saved_exports()
                choices = [f"{e['name']} ({e['mtime'][:10]})" for e in exports]
                return gr.Dropdown(choices=choices)

            def load_export_handler(selection):
                if not selection:
                    return "⚠️ Please select an export file", "", "", ""
                
                exports = get_saved_exports()
                # Find matching export
                for e in exports:
                    if selection.startswith(e["name"]):
                        try:
                            data = load_export_file(e["path"])
                            # Convert to global_data format for visualization
                            vis_data = {
                                "document_id": data.get("document", {}).get("id"),
                                "document_display_name": data.get("document", {}).get("display_name"),
                                "prd_text": data.get("prd_text", ""),
                                "features": data.get("features", []),
                                "test_points": {int(k): v for k, v in data.get("test_points", {}).items()},
                                "test_cases": {}
                            }
                            # Convert test_cases keys
                            for k, v in data.get("test_cases", {}).items():
                                parts = k.split(",")
                                if len(parts) == 2:
                                    vis_data["test_cases"][(int(parts[0]), int(parts[1]))] = v
                            
                            vis = format_data_for_visualization(vis_data)
                            label = format_data_for_labeling(vis_data)
                            json_str = json.dumps(data, ensure_ascii=False, indent=2)
                            return f"✅ Loaded: {e['name']}", vis, label, json_str
                        except Exception as ex:
                            return f"❌ Error loading: {str(ex)}", "", "", ""
                
                return "⚠️ Export file not found", "", "", ""

            export_json_btn.click(
                fn=partial(export_json_handler, global_data),
                outputs=[download_file, save_status, visualization_output, labeling_output, export_output]
            )

            save_server_btn.click(
                fn=partial(save_to_server_handler, global_data),
                outputs=save_status
            )

            refresh_exports_btn.click(
                fn=refresh_exports_handler,
                outputs=saved_exports_dropdown
            )

            load_export_btn.click(
                fn=load_export_handler,
                inputs=saved_exports_dropdown,
                outputs=[save_status, visualization_output, labeling_output, export_output]
            )

        # Event handlers for generation
        def generate_features_handler(backend, doc_choice, requirement):
            # Get PRD text from document or direct input
            if doc_choice:
                doc_id, doc_info = get_document_by_display_name(doc_choice)
                if doc_info:
                    prd_text = doc_info["content"]
                    global_data["document_id"] = doc_id
                    global_data["document_display_name"] = doc_info["display_name"]
                else:
                    yield "⚠️ Document not found", "", gr.Dropdown(choices=[]), gr.Dropdown(choices=[])
                    return
            else:
                yield "⚠️ Please select a document", "", gr.Dropdown(choices=[]), gr.Dropdown(choices=[])
                return

            if not prd_text or not prd_text.strip():
                yield "⚠️ Document content is empty", "", gr.Dropdown(choices=[]), gr.Dropdown(choices=[])
                return

            if backend == "vLLM (Streaming)":
                client = MODEL_CONFIG["vllm"]["client"]
                model_id = MODEL_CONFIG["vllm"]["model_id"]
                if not client:
                    yield "⚠️ Please initialize vLLM first", "", gr.Dropdown(choices=[]), gr.Dropdown(choices=[])
                    return
                for output in generate_features_for_gradio_stream(
                    global_data, client, model_id, prd_text, requirement
                ):
                    yield output
            else:
                llm = MODEL_CONFIG["ollama"]["llm"]
                if not llm:
                    yield "⚠️ Please initialize Ollama first", "", gr.Dropdown(choices=[]), gr.Dropdown(choices=[])
                    return
                result = generate_features_for_gradio(global_data, llm, prd_text, requirement)
                yield result

        gen_feature_btn.click(
            fn=generate_features_handler,
            inputs=[model_backend, prd_doc_dropdown, feature_requirement],
            outputs=[feature_output, feature_thinking, feature_dropdown, feature_dropdown2]
        )

        re_gen_feature_btn.click(
            fn=generate_features_handler,
            inputs=[model_backend, prd_doc_dropdown, feature_requirement],
            outputs=[feature_output, feature_thinking, feature_dropdown, feature_dropdown2]
        )

        def generate_test_points_handler(backend, feature_choice, requirement, current_feature_step3):
            if backend == "vLLM (Streaming)":
                client = MODEL_CONFIG["vllm"]["client"]
                model_id = MODEL_CONFIG["vllm"]["model_id"]
                if not client:
                    yield "⚠️ Please initialize vLLM first", "", gr.Dropdown()
                    return
                for output in generate_test_points_for_gradio_stream(
                    global_data, client, model_id, feature_choice, requirement, current_feature_step3
                ):
                    yield output
            else:
                llm = MODEL_CONFIG["ollama"]["llm"]
                if not llm:
                    yield "⚠️ Please initialize Ollama first", "", gr.Dropdown()
                    return
                result = generate_test_points_for_gradio(
                    global_data, llm, feature_choice, requirement, current_feature_step3
                )
                yield result

        gen_tp_btn.click(
            fn=generate_test_points_handler,
            inputs=[model_backend, feature_dropdown, test_point_requirement, feature_dropdown2],
            outputs=[test_point_output, test_point_thinking, test_point_dropdown]
        )

        regen_tp_btn.click(
            fn=generate_test_points_handler,
            inputs=[model_backend, feature_dropdown, test_point_requirement, feature_dropdown2],
            outputs=[test_point_output, test_point_thinking, test_point_dropdown]
        )

        def generate_test_cases_handler(backend, feature_choice, tp_choice, requirement):
            if backend == "vLLM (Streaming)":
                client = MODEL_CONFIG["vllm"]["client"]
                model_id = MODEL_CONFIG["vllm"]["model_id"]
                if not client:
                    yield "⚠️ Please initialize vLLM first", ""
                    return
                for output in generate_test_cases_for_gradio_stream(
                    global_data, client, model_id, feature_choice, tp_choice, requirement
                ):
                    yield output
            else:
                llm = MODEL_CONFIG["ollama"]["llm"]
                if not llm:
                    yield "⚠️ Please initialize Ollama first", ""
                    return
                result = generate_test_cases_for_gradio(global_data, llm, feature_choice, tp_choice, requirement)
                yield result

        gen_tc_btn.click(
            fn=generate_test_cases_handler,
            inputs=[model_backend, feature_dropdown2, test_point_dropdown, test_case_requirement],
            outputs=[test_case_output, test_case_thinking]
        )

        regen_tc_btn.click(
            fn=generate_test_cases_handler,
            inputs=[model_backend, feature_dropdown2, test_point_dropdown, test_case_requirement],
            outputs=[test_case_output, test_case_thinking]
        )

        # Rating handlers
        def save_feature_rating_handler(score, comment):
            if not global_data.get("features"):
                return "⚠️ No features to rate"
            rating_data = {
                "type": "features",
                "document_id": global_data.get("document_id", ""),
                "item_id": "all_features",
                "item_name": f"{len(global_data['features'])} features",
                "score": int(score),
                "comment": comment
            }
            save_rating(rating_data)
            return f"✅ Rating saved: {int(score)}/5"

        def save_tp_rating_handler(feature_choice, score, comment):
            if not feature_choice:
                return "⚠️ No feature selected"
            feature_id = int(feature_choice.split(".")[0])
            feature_idx = feature_id - 1
            if feature_idx not in global_data.get("test_points", {}):
                return "⚠️ No test points to rate"
            rating_data = {
                "type": "test_points",
                "document_id": global_data.get("document_id", ""),
                "item_id": f"feature_{feature_id}_test_points",
                "item_name": feature_choice,
                "score": int(score),
                "comment": comment
            }
            save_rating(rating_data)
            return f"✅ Rating saved: {int(score)}/5"

        def save_tc_rating_handler(feature_choice, tp_choice, score, comment):
            if not feature_choice or not tp_choice:
                return "⚠️ No test point selected"
            rating_data = {
                "type": "test_cases",
                "document_id": global_data.get("document_id", ""),
                "item_id": f"{feature_choice}_{tp_choice}",
                "item_name": f"{feature_choice} > {tp_choice}",
                "score": int(score),
                "comment": comment
            }
            save_rating(rating_data)
            return f"✅ Rating saved: {int(score)}/5"

        save_feature_rating_btn.click(
            fn=save_feature_rating_handler,
            inputs=[feature_rating, feature_comment],
            outputs=feature_rating_status
        )

        save_tp_rating_btn.click(
            fn=save_tp_rating_handler,
            inputs=[feature_dropdown, tp_rating, tp_comment],
            outputs=tp_rating_status
        )

        save_tc_rating_btn.click(
            fn=save_tc_rating_handler,
            inputs=[feature_dropdown2, test_point_dropdown, tc_rating, tc_comment],
            outputs=tc_rating_status
        )

        # Edit handlers for features
        def load_features_handler():
            if not global_data.get("features"):
                return "[]"
            return json.dumps(global_data["features"], ensure_ascii=False, indent=2)

        def save_features_handler(json_str):
            try:
                features = json.loads(json_str)
                if not isinstance(features, list):
                    return "⚠️ Invalid format: must be a JSON array", gr.Dropdown(), gr.Dropdown()
                global_data["features"] = features
                choices = get_feature_choices_list(global_data)
                return f"✅ Saved {len(features)} features", gr.Dropdown(choices=choices), gr.Dropdown(choices=choices)
            except json.JSONDecodeError as e:
                return f"⚠️ JSON parse error: {str(e)}", gr.Dropdown(), gr.Dropdown()

        load_features_btn.click(fn=load_features_handler, outputs=feature_edit_json)
        save_features_btn.click(
            fn=save_features_handler,
            inputs=feature_edit_json,
            outputs=[feature_edit_status, feature_dropdown, feature_dropdown2]
        )

        # Edit handlers for test points
        def load_tp_handler(feature_choice):
            if not feature_choice:
                return "[]"
            feature_id = int(feature_choice.split(".")[0])
            feature_idx = feature_id - 1
            test_points = global_data.get("test_points", {}).get(feature_idx, [])
            return json.dumps(test_points, ensure_ascii=False, indent=2)

        def save_tp_handler(feature_choice, json_str, current_feature_step3):
            if not feature_choice:
                return "⚠️ No feature selected", gr.Dropdown()
            try:
                test_points = json.loads(json_str)
                if not isinstance(test_points, list):
                    return "⚠️ Invalid format: must be a JSON array", gr.Dropdown()
                feature_id = int(feature_choice.split(".")[0])
                feature_idx = feature_id - 1
                global_data["test_points"][feature_idx] = test_points
                # Update test point dropdown if same feature selected in Step 3
                if current_feature_step3 and current_feature_step3 == feature_choice:
                    tp_choices = get_test_point_choices(global_data, feature_choice)
                    return f"✅ Saved {len(test_points)} test points", gr.Dropdown(choices=tp_choices)
                return f"✅ Saved {len(test_points)} test points", gr.Dropdown()
            except json.JSONDecodeError as e:
                return f"⚠️ JSON parse error: {str(e)}", gr.Dropdown()

        load_tp_btn.click(fn=load_tp_handler, inputs=feature_dropdown, outputs=tp_edit_json)
        save_tp_btn.click(
            fn=save_tp_handler,
            inputs=[feature_dropdown, tp_edit_json, feature_dropdown2],
            outputs=[tp_edit_status, test_point_dropdown]
        )

        # Edit handlers for test cases
        def load_tc_handler(feature_choice, tp_choice):
            if not feature_choice or not tp_choice:
                return "[]"
            feature_id = int(feature_choice.split(".")[0])
            tp_id = int(tp_choice.split(".")[0])
            feature_idx = feature_id - 1
            tp_idx = tp_id - 1
            test_cases = global_data.get("test_cases", {}).get((feature_idx, tp_idx), [])
            return json.dumps(test_cases, ensure_ascii=False, indent=2)

        def save_tc_handler(feature_choice, tp_choice, json_str):
            if not feature_choice or not tp_choice:
                return "⚠️ No test point selected"
            try:
                test_cases = json.loads(json_str)
                if not isinstance(test_cases, list):
                    return "⚠️ Invalid format: must be a JSON array"
                feature_id = int(feature_choice.split(".")[0])
                tp_id = int(tp_choice.split(".")[0])
                feature_idx = feature_id - 1
                tp_idx = tp_id - 1
                global_data["test_cases"][(feature_idx, tp_idx)] = test_cases
                return f"✅ Saved {len(test_cases)} test cases"
            except json.JSONDecodeError as e:
                return f"⚠️ JSON parse error: {str(e)}"

        load_tc_btn.click(fn=load_tc_handler, inputs=[feature_dropdown2, test_point_dropdown], outputs=tc_edit_json)
        save_tc_btn.click(fn=save_tc_handler, inputs=[feature_dropdown2, test_point_dropdown, tc_edit_json], outputs=tc_edit_status)

    return demo


if __name__ == "__main__":
    for proxy_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
        proxy_value = os.environ.get(proxy_var)
        if proxy_value and not proxy_value.startswith(('http://', 'https://', 'socks5://')):
            os.environ[proxy_var] = f'http://{proxy_value}'

    no_proxy = os.environ.get('NO_PROXY', os.environ.get('no_proxy', ''))
    if no_proxy:
        os.environ['NO_PROXY'] = no_proxy + ',localhost,127.0.0.1,0.0.0.0'
        os.environ['no_proxy'] = no_proxy + ',localhost,127.0.0.1,0.0.0.0'
    else:
        os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'
        os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'

    demo = init_gradio_page()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7868,
        share=False
    )
