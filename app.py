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
    get_test_point_choices,
    generate_ui_automation_for_gradio, generate_ui_automation_for_gradio_stream,
    DEFAULT_JSONL_PATH
)
from export_util import (
    export_to_json_string, save_to_server, get_saved_exports,
    format_data_for_visualization, format_data_for_labeling, load_export_file
)
from rating_util import save_rating, get_ratings, get_rating_summary, export_ratings_csv
from rag_util import (
    retrieve_knowledge, format_retrieved_content, check_index_exists,
    build_index_from_docx, build_index_from_docx_with_mode, get_index_stats,
    chunk_docx_with_llm, retrieve_jsonl_examples, format_jsonl_examples_for_prompt,
    configure_embeddings, get_embedding_config
)

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

# Available Ollama embedding models
OLLAMA_EMBEDDING_MODELS = ["bge-large", "bge-m3"]

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
                    with gr.Row():
                        vllm_host = gr.Textbox(
                            label="vLLM Host",
                            value="localhost",
                            placeholder="localhost or IP address"
                        )
                        vllm_port = gr.Number(
                            label="Port",
                            value=12349,
                            precision=0
                        )
                    vllm_model = gr.Textbox(
                        label="Model ID",
                        value="Qwen3-8B",
                        placeholder="Qwen3-8B"
                    )
                    init_vllm_btn = gr.Button("🚀 Initialize vLLM", variant="primary")
                with gr.Column(visible=False) as ollama_config:
                    with gr.Row():
                        ollama_host = gr.Textbox(
                            label="Ollama Host",
                            value="localhost",
                            placeholder="localhost or IP address"
                        )
                        ollama_port = gr.Number(
                            label="Port",
                            value=11434,
                            precision=0
                        )
                    ollama_model = gr.Textbox(
                        label="Ollama Model",
                        value="qwen3:8b",
                        placeholder="qwen3:8b"
                    )
                    init_ollama_btn = gr.Button("🚀 Initialize Ollama", variant="primary")

            model_status = gr.Textbox(label="Model Status", value="⏳ Model not initialized", interactive=False)
        
        # Embedding Configuration Section
        with gr.Accordion("🔤 Embedding Configuration", open=True):
            gr.Markdown("Configure embedding model for RAG search. Supports local HuggingFace model or Ollama API.")
            with gr.Row():
                embedding_mode = gr.Radio(
                    label="Embedding Mode",
                    choices=["Local (HuggingFace)", "API (Ollama)"],
                    value="API (Ollama)"
                )
            
            with gr.Row():
                with gr.Column(visible=False) as embed_local_config:
                    embed_local_path = gr.Textbox(
                        label="Local Model Path",
                        value="/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/chensenda/codes/models/bge-large-zh-v1.5",
                        placeholder="Path to local HuggingFace embedding model"
                    )
                
                with gr.Column(visible=True) as embed_api_config:
                    with gr.Row():
                        embed_api_host = gr.Textbox(
                            label="Ollama Host",
                            value="localhost",
                            placeholder="localhost or IP address"
                        )
                        embed_api_port = gr.Number(
                            label="Port",
                            value=11434,
                            precision=0
                        )
                    embed_api_model = gr.Dropdown(
                        label="Embedding Model",
                        choices=OLLAMA_EMBEDDING_MODELS,
                        value="bge-large",
                        allow_custom_value=True
                    )
            
            init_embedding_btn = gr.Button("🚀 Initialize Embedding", variant="primary")
            embedding_status = gr.Textbox(label="Embedding Status", value="⏳ Using Ollama API embedding (bge-large)", interactive=False)
            
            def toggle_embedding_config(mode):
                if mode == "Local (HuggingFace)":
                    return gr.Column(visible=True), gr.Column(visible=False)
                else:
                    return gr.Column(visible=False), gr.Column(visible=True)
            
            embedding_mode.change(
                fn=toggle_embedding_config,
                inputs=embedding_mode,
                outputs=[embed_local_config, embed_api_config]
            )
            
            def init_embedding_handler(mode, local_path, api_host, api_port, api_model):
                """Initialize embedding model based on selected mode"""
                try:
                    if mode == "Local (HuggingFace)":
                        configure_embeddings(
                            mode="local",
                            local_model_path=local_path
                        )
                        return f"✅ Local embedding initialized: {local_path.split('/')[-1]}"
                    else:
                        api_port_int = int(api_port)
                        base_url = f"http://{api_host}:{api_port_int}"
                        configure_embeddings(
                            mode="api",
                            api_base_url=base_url,
                            api_model=api_model
                        )
                        return f"✅ Ollama embedding initialized: {api_model} @ {base_url}"
                except Exception as e:
                    return f"❌ Error: {str(e)}"
            
            init_embedding_btn.click(
                fn=init_embedding_handler,
                inputs=[embedding_mode, embed_local_path, embed_api_host, embed_api_port, embed_api_model],
                outputs=embedding_status
            )

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

            def init_vllm_with_host_port(host, port, model_id):
                """Initialize vLLM with host and port"""
                port_int = int(port)
                base_url = f"http://{host}:{port_int}/v1"
                return init_vllm(base_url, model_id)
            
            init_vllm_btn.click(
                fn=init_vllm_with_host_port,
                inputs=[vllm_host, vllm_port, vllm_model],
                outputs=model_status
            )

            def init_ollama_with_config(host, port, model_name):
                """Initialize Ollama with host, port and model"""
                port_int = int(port)
                base_url = f"http://{host}:{port_int}"
                MODEL_CONFIG["ollama"]["model"] = model_name
                MODEL_CONFIG["ollama"]["llm"] = OllamaLLM(
                    model=model_name,
                    base_url=base_url
                )
                return f"✅ Ollama initialized (Host: {host}:{port_int}, Model: {model_name})"

            init_ollama_btn.click(
                fn=init_ollama_with_config,
                inputs=[ollama_host, ollama_port, ollama_model],
                outputs=model_status
            )

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
                    
                    # RAG Index Building Section
                    gr.Markdown("### 🔧 RAG Index")
                    index_stats = get_index_stats()
                    if index_stats['exists']:
                        count_str = str(index_stats['num_documents']) if index_stats['num_documents'] >= 0 else "?"
                        init_status = f"✅ Index exists ({count_str} chunks)"
                    else:
                        init_status = "❌ No index"
                    index_status = gr.Textbox(
                        label="Index Status",
                        value=init_status,
                        interactive=False,
                        lines=1
                    )
                    
                    # Chunking mode selection
                    chunk_mode = gr.Radio(
                        label="Chunking Mode",
                        choices=["⚡ Fast (Rule-based)", "🤖 LLM (Semantic)"],
                        value="⚡ Fast (Rule-based)"
                    )
                    
                    with gr.Row():
                        preview_chunks_btn = gr.Button("👁️ Preview Chunks", variant="secondary")
                        build_index_btn = gr.Button("🔨 Build Index", variant="primary")
                    
                    index_progress = gr.Textbox(
                        label="Progress",
                        value="",
                        interactive=False,
                        lines=2
                    )
                    
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("📄 Document Content"):
                            doc_preview = gr.Textbox(
                                label="Document Content",
                                value=initial_preview,
                                lines=25,
                                max_lines=1000,
                                interactive=False,
                                show_copy_button=True
                            )
                        with gr.Tab("🧩 Chunk Preview"):
                            chunk_preview_info = gr.Markdown("Click **Preview Chunks** to see chunking results")
                            chunk_preview = gr.Dataframe(
                                headers=["#", "Size", "Preview"],
                                datatype=["number", "number", "str"],
                                row_count=10,
                                col_count=(3, "fixed"),
                                interactive=False,
                                wrap=True
                            )
            
            # Chunk preview handler
            def preview_chunks_handler(doc_choice, mode):
                if not doc_choice:
                    return "⚠️ Please select a document first", []
                
                doc_id, doc_info = get_document_by_display_name(doc_choice)
                if not doc_info:
                    return "❌ Document not found", []
                
                docx_path = doc_info.get("path")
                if not docx_path or not os.path.exists(docx_path):
                    return "❌ Document file not found", []
                
                try:
                    use_llm = "LLM" in mode
                    chunks = chunk_docx_with_llm(docx_path, use_llm=use_llm)
                    
                    # Format for dataframe
                    data = []
                    for i, chunk in enumerate(chunks):
                        content = chunk.get("content", "")
                        preview_text = content[:150].replace("\n", " ") + ("..." if len(content) > 150 else "")
                        data.append([i + 1, len(content), preview_text])
                    
                    mode_name = "LLM" if use_llm else "Fast"
                    info = f"### ✅ {len(chunks)} chunks ({mode_name} mode)\n\nTotal characters: {sum(len(c.get('content', '')) for c in chunks)}"
                    return info, data
                except Exception as e:
                    return f"❌ Error: {str(e)}", []
            
            preview_chunks_btn.click(
                fn=preview_chunks_handler,
                inputs=[uploaded_doc_dropdown, chunk_mode],
                outputs=[chunk_preview_info, chunk_preview]
            )
            
            # Index building handler
            def build_index_handler(doc_choice, mode):
                if not doc_choice:
                    yield "❌ Please select a document first", "❌ No document selected"
                    return
                
                doc_id, doc_info = get_document_by_display_name(doc_choice)
                if not doc_info:
                    yield "❌ Document not found", "❌ Document not found"
                    return
                
                docx_path = doc_info.get("path")
                if not docx_path or not os.path.exists(docx_path):
                    yield "❌ Document file not found", "❌ File not found"
                    return
                
                use_llm = "LLM" in mode
                mode_name = "LLM" if use_llm else "Fast"
                
                progress_messages = []
                def progress_callback(msg):
                    progress_messages.append(msg)
                
                yield f"🔄 Building index ({mode_name} mode)...", "🔄 Processing..."
                
                try:
                    success = build_index_from_docx_with_mode(docx_path, "faiss_index", use_llm=use_llm, progress_callback=progress_callback)
                    if success:
                        stats = get_index_stats()
                        count_str = str(stats['num_documents']) if stats['num_documents'] >= 0 else "?"
                        final_status = f"✅ Index exists ({count_str} chunks)"
                        yield "\n".join(progress_messages[-3:]) if progress_messages else "✅ Done!", final_status
                    else:
                        yield "❌ Failed to build index", "❌ Build failed"
                except Exception as e:
                    yield f"❌ Error: {str(e)}", "❌ Error"
            
            build_index_btn.click(
                fn=build_index_handler,
                inputs=[uploaded_doc_dropdown, chunk_mode],
                outputs=[index_progress, index_status]
            )

        # Step 1: PRD -> Features
        with gr.Tab("📄 Step 1: PRD → Features"):
            with gr.Row():
                with gr.Column(scale=2):
                    # PRD Source Mode Selection
                    prd_source_mode = gr.Radio(
                        label="PRD Source",
                        choices=["📁 Document", "🔍 RAG Search"],
                        value="📁 Document"
                    )
                    
                    # Document Mode Components
                    with gr.Column(visible=True) as doc_mode_col:
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
                    
                    # RAG Mode Components
                    with gr.Column(visible=False) as rag_mode_col:
                        rag_query = gr.Textbox(
                            label="RAG Query",
                            placeholder="Enter keywords or questions to search PRD knowledge base...",
                            lines=2
                        )
                        with gr.Row():
                            rag_top_k = gr.Slider(
                                minimum=1, maximum=20, step=1, value=7,
                                label="Top K Results"
                            )
                            rag_search_btn = gr.Button("🔍 Search", variant="secondary")
                        rag_status = gr.Textbox(
                            label="RAG Status",
                            value="⏳ Enter query and click Search",
                            interactive=False,
                            lines=1
                        )
                        rag_content_preview = gr.Textbox(
                            label="Retrieved PRD Content",
                            placeholder="Retrieved content will appear here...",
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
            
            # PRD Source Mode Toggle
            def toggle_prd_source_mode(mode):
                if mode == "📁 Document":
                    return gr.Column(visible=True), gr.Column(visible=False)
                else:
                    return gr.Column(visible=False), gr.Column(visible=True)
            
            prd_source_mode.change(
                fn=toggle_prd_source_mode,
                inputs=prd_source_mode,
                outputs=[doc_mode_col, rag_mode_col]
            )
            
            # RAG Search Handler
            def rag_search_handler(query, top_k):
                if not query or not query.strip():
                    return "⚠️ Please enter a query", ""
                
                if not check_index_exists("faiss_index"):
                    return "❌ FAISS index not found. Please build index first.", ""
                
                try:
                    fragments = retrieve_knowledge(query, top_k=int(top_k), use_reranker=False)
                    if not fragments:
                        return "⚠️ No results found", ""
                    
                    content = format_retrieved_content(fragments, query)
                    return f"✅ Retrieved {len(fragments)} fragments", content
                except Exception as e:
                    return f"❌ Error: {str(e)}", ""
            
            rag_search_btn.click(
                fn=rag_search_handler,
                inputs=[rag_query, rag_top_k],
                outputs=[rag_status, rag_content_preview]
            )

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
            with gr.Tabs():
                # Sub-tab 1: Traditional Test Cases
                with gr.Tab("📋 Traditional Test Cases"):
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

                # Sub-tab 2: UI Automation Test Cases
                with gr.Tab("🤖 UI Automation Test Cases"):
                    gr.Markdown("""
                    ### Generate UI Automation Test Steps
                    Generate step-by-step UI automation test sequences (CLICK, SCROLL, TEXT, COMPLETE) 
                    for mobile app testing. Uses RAG to retrieve similar examples for few-shot learning.
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            # Input section
                            ui_prd_input = gr.Textbox(
                                label="📄 PRD Document",
                                placeholder="Paste the PRD document content here, or it will use the current document...",
                                lines=6,
                                max_lines=20
                            )
                            
                            with gr.Row():
                                with gr.Column():
                                    ui_feature_input = gr.Textbox(
                                        label="🎯 Feature",
                                        placeholder="E.g., Cross-app location search and ride-hailing integration",
                                        lines=2
                                    )
                                with gr.Column():
                                    ui_testpoint_input = gr.Textbox(
                                        label="📌 Test Point",
                                        placeholder="E.g., Verify user can search 'grocery store' in Maps and book Uber ride to selected location",
                                        lines=2
                                    )
                            
                            ui_testcase_name_input = gr.Textbox(
                                label="📝 Test Case Name",
                                placeholder="E.g., Search nearby grocery store in Maps and book Uber ride to the location",
                                lines=1
                            )
                            
                            # RAG settings
                            with gr.Accordion("🔍 RAG Settings", open=True):
                                ui_use_rag = gr.Checkbox(
                                    label="Enable RAG (Retrieve similar examples)",
                                    value=True
                                )
                                ui_rag_topk = gr.Slider(
                                    minimum=1, maximum=5, step=1, value=3,
                                    label="Number of Examples to Retrieve"
                                )
                                ui_jsonl_path = gr.Textbox(
                                    label="JSONL Knowledge Base Path",
                                    value=DEFAULT_JSONL_PATH,
                                    lines=1
                                )
                                with gr.Row():
                                    preview_rag_btn = gr.Button("🔍 Preview RAG Results", variant="secondary")
                                ui_rag_status = gr.Textbox(
                                    label="RAG Status",
                                    value="",
                                    interactive=False,
                                    lines=1
                                )
                                ui_rag_preview = gr.Markdown(
                                    label="Retrieved Examples Preview",
                                    value="*Click 'Preview RAG Results' to see retrieved examples*"
                                )
                            
                            ui_additional_req = gr.Textbox(
                                label="Additional Requirements (Optional)",
                                placeholder="E.g., Include error handling steps, use specific app names...",
                                lines=2
                            )
                            
                            with gr.Row():
                                gen_ui_auto_btn = gr.Button("✨ Generate UI Automation Steps", variant="primary")
                                regen_ui_auto_btn = gr.Button("🔄 Regenerate")
                            
                            # Quick fill from current data
                            with gr.Accordion("📥 Quick Fill from Current Data", open=False):
                                gr.Markdown("Fill inputs from currently selected feature and test point")
                                with gr.Row():
                                    ui_feature_select = gr.Dropdown(
                                        label="Select Feature",
                                        choices=[],
                                        interactive=True
                                    )
                                    ui_tp_select = gr.Dropdown(
                                        label="Select Test Point",
                                        choices=[],
                                        interactive=True
                                    )
                                with gr.Row():
                                    refresh_ui_feature_btn = gr.Button("🔄 Refresh")
                                    fill_from_selection_btn = gr.Button("📥 Fill Inputs", variant="secondary")
                        
                        with gr.Column(scale=3):
                            ui_auto_output = gr.Markdown(label="Generated UI Automation Steps")
                            ui_auto_thinking = gr.Textbox(label="Status", lines=2)
                            
                            # JSON output for copy
                            with gr.Accordion("📋 JSON Output (Copy)", open=True):
                                ui_auto_json = gr.Code(
                                    label="UI Automation Steps JSON",
                                    language="json",
                                    lines=15
                                )
                                copy_json_btn = gr.Button("📋 Copy to Clipboard", size="sm")

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
            
            # UI Automation Tab Event Handlers
            refresh_ui_feature_btn.click(
                fn=partial(update_feature_dropdown, global_data),
                outputs=ui_feature_select
            )
            
            ui_feature_select.change(
                fn=partial(update_test_point_dropdown, global_data),
                inputs=ui_feature_select,
                outputs=ui_tp_select
            )
            
            def fill_ui_inputs_from_selection(feature_choice, tp_choice):
                """Fill UI automation inputs from selected feature and test point"""
                prd_text = global_data.get("prd_text", "")
                
                feature_text = ""
                if feature_choice:
                    feature_id = int(feature_choice.split(".")[0])
                    feature_idx = feature_id - 1
                    if 0 <= feature_idx < len(global_data.get("features", [])):
                        feature = global_data["features"][feature_idx]
                        feature_text = f"{feature.get('name', '')}: {feature.get('description', '')}"
                
                tp_text = ""
                if tp_choice and feature_choice:
                    feature_id = int(feature_choice.split(".")[0])
                    tp_id = int(tp_choice.split(".")[0])
                    feature_idx = feature_id - 1
                    tp_idx = tp_id - 1
                    if feature_idx in global_data.get("test_points", {}):
                        tps = global_data["test_points"][feature_idx]
                        if 0 <= tp_idx < len(tps):
                            tp = tps[tp_idx]
                            tp_text = f"{tp.get('name', '')}: {tp.get('description', '')}"
                
                return prd_text, feature_text, tp_text
            
            fill_from_selection_btn.click(
                fn=fill_ui_inputs_from_selection,
                inputs=[ui_feature_select, ui_tp_select],
                outputs=[ui_prd_input, ui_feature_input, ui_testpoint_input]
            )
            
            def preview_rag_results(feature_text, tp_text, rag_topk, jsonl_path):
                """Preview RAG retrieved examples"""
                if not feature_text and not tp_text:
                    return "⚠️ Please enter feature or test point first", "*No query provided*"
                
                query = f"{feature_text} {tp_text}".strip()
                if not query:
                    return "⚠️ Query is empty", "*No query provided*"
                
                try:
                    examples = retrieve_jsonl_examples(query, jsonl_path, top_k=int(rag_topk))
                    if not examples:
                        return "⚠️ No examples found", "*No matching examples in knowledge base*"
                    
                    # Format for display
                    formatted = format_jsonl_examples_for_prompt(examples)
                    status = f"✅ Retrieved {len(examples)} examples"
                    return status, formatted
                except Exception as e:
                    return f"❌ Error: {str(e)}", f"*Error retrieving examples: {str(e)}*"
            
            preview_rag_btn.click(
                fn=preview_rag_results,
                inputs=[ui_feature_input, ui_testpoint_input, ui_rag_topk, ui_jsonl_path],
                outputs=[ui_rag_status, ui_rag_preview]
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
        def generate_features_handler(backend, source_mode, doc_choice, rag_content, requirement):
            # Get PRD text based on source mode
            if source_mode == "📁 Document":
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
            else:  # RAG Mode
                if not rag_content or not rag_content.strip():
                    yield "⚠️ Please search and retrieve content first", "", gr.Dropdown(choices=[]), gr.Dropdown(choices=[])
                    return
                prd_text = rag_content
                global_data["document_id"] = "rag_retrieved"
                global_data["document_display_name"] = "RAG Retrieved Content"

            if not prd_text or not prd_text.strip():
                yield "⚠️ PRD content is empty", "", gr.Dropdown(choices=[]), gr.Dropdown(choices=[])
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
            inputs=[model_backend, prd_source_mode, prd_doc_dropdown, rag_content_preview, feature_requirement],
            outputs=[feature_output, feature_thinking, feature_dropdown, feature_dropdown2]
        )

        re_gen_feature_btn.click(
            fn=generate_features_handler,
            inputs=[model_backend, prd_source_mode, prd_doc_dropdown, rag_content_preview, feature_requirement],
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

        # UI Automation Test Case Generation Handlers
        def generate_ui_automation_handler(backend, prd_text, feature_text, tp_text, tc_name,
                                          use_rag, rag_topk, jsonl_path, additional_req):
            """Handler for UI automation test case generation"""
            # Use current PRD if not provided
            if not prd_text or not prd_text.strip():
                prd_text = global_data.get("prd_text", "")
            
            if not prd_text or not prd_text.strip():
                yield "⚠️ Please provide PRD document content", "", "", "", ""
                return
            
            if not feature_text or not feature_text.strip():
                yield "⚠️ Please provide feature description", "", "", "", ""
                return
            
            if not tp_text or not tp_text.strip():
                yield "⚠️ Please provide test point description", "", "", "", ""
                return
            
            # First, retrieve and show RAG examples if enabled
            rag_status = ""
            rag_preview = ""
            rag_examples_text = ""
            if use_rag:
                yield "🔍 Retrieving reference examples...", "", "", "🔄 Searching...", ""
                query = f"{feature_text} {tp_text} {tc_name or ''}".strip()
                try:
                    examples = retrieve_jsonl_examples(query, jsonl_path, top_k=int(rag_topk))
                    if examples:
                        rag_status = f"✅ Retrieved {len(examples)} reference examples"
                        rag_preview = format_jsonl_examples_for_prompt(examples)
                        rag_examples_text = rag_preview
                    else:
                        rag_status = "⚠️ No matching examples found"
                        rag_preview = "*No matching examples in knowledge base*"
                except Exception as e:
                    rag_status = f"❌ RAG Error: {str(e)}"
                    rag_preview = f"*Error: {str(e)}*"
            else:
                rag_status = "ℹ️ RAG disabled"
                rag_preview = "*RAG is disabled*"
            
            # Show RAG results
            yield "🔄 Starting generation...", "", "", rag_status, rag_preview
            
            if backend == "vLLM (Streaming)":
                client = MODEL_CONFIG["vllm"]["client"]
                model_id = MODEL_CONFIG["vllm"]["model_id"]
                if not client:
                    yield "⚠️ Please initialize vLLM first", "", "", rag_status, rag_preview
                    return
                for output in generate_ui_automation_for_gradio_stream(
                    global_data, client, model_id,
                    prd_text, feature_text, tp_text, tc_name,
                    use_rag=False,  # Already retrieved above
                    rag_top_k=int(rag_topk),
                    additional_requirement=additional_req,
                    jsonl_path=jsonl_path,
                    rag_examples_text=rag_examples_text if use_rag else None
                ):
                    md_output, status, steps = output
                    json_str = json.dumps(steps, ensure_ascii=False, indent=2) if steps else ""
                    yield md_output, status, json_str, rag_status, rag_preview
            else:
                llm = MODEL_CONFIG["ollama"]["llm"]
                if not llm:
                    yield "⚠️ Please initialize Ollama first", "", "", rag_status, rag_preview
                    return
                output, status, steps = generate_ui_automation_for_gradio(
                    global_data, llm,
                    prd_text, feature_text, tp_text, tc_name,
                    use_rag=use_rag, rag_top_k=int(rag_topk),
                    additional_requirement=additional_req,
                    jsonl_path=jsonl_path
                )
                json_str = json.dumps(steps, ensure_ascii=False, indent=2) if steps else ""
                yield output, status, json_str, rag_status, rag_preview

        gen_ui_auto_btn.click(
            fn=generate_ui_automation_handler,
            inputs=[model_backend, ui_prd_input, ui_feature_input, ui_testpoint_input, ui_testcase_name_input,
                   ui_use_rag, ui_rag_topk, ui_jsonl_path, ui_additional_req],
            outputs=[ui_auto_output, ui_auto_thinking, ui_auto_json, ui_rag_status, ui_rag_preview]
        )

        regen_ui_auto_btn.click(
            fn=generate_ui_automation_handler,
            inputs=[model_backend, ui_prd_input, ui_feature_input, ui_testpoint_input, ui_testcase_name_input,
                   ui_use_rag, ui_rag_topk, ui_jsonl_path, ui_additional_req],
            outputs=[ui_auto_output, ui_auto_thinking, ui_auto_json, ui_rag_status, ui_rag_preview]
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
