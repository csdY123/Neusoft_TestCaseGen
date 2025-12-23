from gradio_client import Client
import json
import re
import os

def extract_json_from_markdown(markdown_text):
    """Extract JSON from markdown code block"""
    if not markdown_text:
        return None
    
    # Pattern 1: ```json ... ``` (most common format)
    json_pattern = r'```json\s*\n(.*?)```'
    matches = re.findall(json_pattern, markdown_text, re.DOTALL)
    if matches:
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    # Pattern 2: ``` ... ``` (without language tag, but contains JSON)
    code_pattern = r'```\s*\n(.*?)```'
    matches = re.findall(code_pattern, markdown_text, re.DOTALL)
    for match in matches:
        match = match.strip()
        if match.startswith('[') or match.startswith('{'):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # Pattern 3: Look for JSON after "**JSON Output**:" or "JSON Output:"
    # This handles the format shown in terminal selection
    json_markers = ['**JSON Output**:', 'JSON Output:', '**JSON Output**']
    for marker in json_markers:
        if marker in markdown_text:
            # Find the marker and look for code block after it
            marker_idx = markdown_text.find(marker)
            if marker_idx != -1:
                # Look for code block after marker
                remaining = markdown_text[marker_idx + len(marker):]
                # Try to find ```json or ``` code block
                code_block_pattern = r'```(?:json)?\s*\n(.*?)```'
                code_matches = re.findall(code_block_pattern, remaining, re.DOTALL)
                if code_matches:
                    for code_match in code_matches:
                        try:
                            return json.loads(code_match.strip())
                        except json.JSONDecodeError:
                            continue
                
                # If no code block, try to extract JSON directly
                for char in ['[', '{']:
                    idx = remaining.find(char)
                    if idx != -1:
                        # Try to extract complete JSON by matching brackets
                        bracket_count = 0
                        start_idx = idx
                        for i in range(idx, len(remaining)):
                            if remaining[i] == char:
                                bracket_count += 1
                            elif remaining[i] == ('}' if char == '{' else ']'):
                                bracket_count -= 1
                                if bracket_count == 0:
                                    try:
                                        json_str = remaining[start_idx:i+1]
                                        return json.loads(json_str)
                                    except json.JSONDecodeError:
                                        break
                        break
    
    return None

def test_ui_automation_generation():
    """Test UI automation test case generation via Gradio API"""
    
    # Initialize Gradio client
    client = Client("http://localhost:7868/")
    
    # Test parameters
    test_params = {
        "backend": "vLLM (Streaming)",
        "prd_text": "需求背景：用户在使用地图或本地服务应用时，常需要保存感兴趣的地点以便后续查看或导航。功能概述：应用应支持用户通过关键词搜索地点，在搜索结果列表中选择具体地点后，提供收藏（Save）功能，将该地点加入用户的收藏夹。用户场景：用户打开应用，点击搜索框，输入'KFC'并执行搜索，在返回的结果列表中点击第一条地点详情，随后点击Save按钮，系统应成功保存该地点至用户收藏。",
        "feature_text": "搜索结果的收藏功能",
        "tp_text": "搜索KFC并收藏搜索结果中的第一条地点",
        "tc_name": "验证用户在搜索KFC后，点击搜索结果列表中的第一条地点，能够成功点击Save按钮完成收藏操作",
        "use_rag": True,
        "rag_topk": 3,
        "jsonl_path": "/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/chensenda/codes/Neusoft/TestCaseGen/update_jsonl",
        "additional_req": "Hello!!",
        "api_name": "/generate_ui_automation_handler"
    }
    
    print("=" * 80)
    print("Testing UI Automation Test Case Generation")
    print("=" * 80)
    print(f"\nTest Parameters:")
    print(f"  Backend: {test_params['backend']}")
    print(f"  Feature: {test_params['feature_text']}")
    print(f"  Test Point: {test_params['tp_text']}")
    print(f"  Use RAG: {test_params['use_rag']}")
    print(f"  RAG Top-K: {test_params['rag_topk']}")
    print(f"  JSONL Path: {test_params['jsonl_path']}")
    print("\n" + "=" * 80)
    print("Calling API...")
    print("=" * 80 + "\n")
    
    try:
        # Call the API
        result = client.predict(**test_params)
        
        # Parse results
        # [0] Generated UI Automation Steps (Markdown)
        # [1] Status (Textbox)
        # [2] UI Automation Steps JSON (Code)
        # [3] RAG Status (Textbox)
        # [4] Retrieved Examples Preview (Markdown)
        
        generated_steps = result[0] if len(result) > 0 else ""
        status = result[1] if len(result) > 1 else ""
        steps_json = result[2] if len(result) > 2 else ""
        rag_status = result[3] if len(result) > 3 else ""
        rag_examples = result[4] if len(result) > 4 else ""
        
        # Display results
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        
        print("\n[1] STATUS:")
        print("-" * 80)
        print(status)
        
        print("\n[2] RAG STATUS:")
        print("-" * 80)
        print(rag_status)
        
        print("\n[3] GENERATED UI AUTOMATION STEPS:")
        print("-" * 80)
        print(generated_steps)
        
        print("\n[4] UI AUTOMATION STEPS JSON:")
        print("-" * 80)
        if steps_json:
            try:
                # Try to pretty print JSON if it's valid JSON
                json_data = json.loads(steps_json)
                print(json.dumps(json_data, ensure_ascii=False, indent=2))
            except (json.JSONDecodeError, TypeError):
                # If not valid JSON, print as is
                print(steps_json)
        else:
            print("(Empty)")
        
        print("\n[5] RETRIEVED EXAMPLES PREVIEW:")
        print("-" * 80)
        print(rag_examples if rag_examples else "(No examples retrieved)")
        
        print("\n" + "=" * 80)
        print("Test completed successfully!")
        print("=" * 80)
        
        return {
            "success": True,
            "generated_steps": generated_steps,
            "status": status,
            "steps_json": steps_json,
            "rag_status": rag_status,
            "rag_examples": rag_examples
        }
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


def batch_test_jsonl(input_jsonl_path, output_jsonl_path=None, backend="vLLM (Streaming)", 
                     use_rag=True, rag_topk=3, jsonl_path=None, additional_req=""):
    """
    Batch test jsonl file and add pred field to each record
    
    Args:
        input_jsonl_path: Path to input jsonl file
        output_jsonl_path: Path to output jsonl file (if None, auto-generate)
        backend: Backend to use
        use_rag: Whether to use RAG
        rag_topk: RAG top-k value
        jsonl_path: RAG jsonl path
        additional_req: Additional requirements
    """
    # Initialize Gradio client
    client = Client("http://localhost:7868/")
    
    # Auto-generate output path if not provided
    if output_jsonl_path is None:
        base_name = os.path.splitext(os.path.basename(input_jsonl_path))[0]
        output_dir = os.path.dirname(input_jsonl_path)
        output_jsonl_path = os.path.join(output_dir, f"{base_name}_with_pred.jsonl")
    
    # Read input jsonl
    print(f"Reading input file: {input_jsonl_path}")
    records = []
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    
    print(f"Total records: {len(records)}")
    print(f"Output file: {output_jsonl_path}")
    print("=" * 80)
    
    # Open output file for writing (append mode, but we'll create new file)
    # Use 'w' mode to create new file, then switch to append if needed
    output_file = open(output_jsonl_path, 'w', encoding='utf-8')
    
    try:
        # Process each record and write immediately
        processed_count = 0
        for idx, record in enumerate(records, 1):
            print(f"\n[{idx}/{len(records)}] Processing episode_id: {record.get('episode_id', 'unknown')}")
            
            # Extract prd_info
            prd_info = record.get('prd_info', {})
            prd_text = prd_info.get('prd_document', '')
            feature_text = prd_info.get('feature', '')
            tp_text = prd_info.get('test_point', '')
            tc_name = prd_info.get('test_case_name', '')
            
            # Prepare test parameters
            test_params = {
                "backend": backend,
                "prd_text": prd_text,
                "feature_text": feature_text,
                "tp_text": tp_text,
                "tc_name": tc_name,
                "use_rag": use_rag,
                "rag_topk": rag_topk,
                "jsonl_path": jsonl_path or "",
                "additional_req": additional_req,
                "api_name": "/generate_ui_automation_handler"
            }
            
            try:
                # Call the API
                result = client.predict(**test_params)
                
                # Get generated_steps (result[0])
                generated_steps = result[0] if len(result) > 0 else ""
                
                # Extract JSON from generated_steps
                pred_json = extract_json_from_markdown(generated_steps)
                
                # If extraction failed, try to use steps_json (result[2])
                if pred_json is None:
                    steps_json = result[2] if len(result) > 2 else ""
                    if steps_json:
                        try:
                            pred_json = json.loads(steps_json)
                        except (json.JSONDecodeError, TypeError):
                            pass
                
                # Add pred field to record
                record_copy = record.copy()
                record_copy['pred'] = pred_json if pred_json is not None else []
                
                # Write immediately to file
                output_file.write(json.dumps(record_copy, ensure_ascii=False) + '\n')
                output_file.flush()  # Ensure data is written to disk immediately
                processed_count += 1
                
                if pred_json:
                    print(f"  ✓ Successfully extracted {len(pred_json)} steps and written to file")
                else:
                    print(f"  ⚠ Warning: Failed to extract JSON, pred set to empty list (written to file)")
                    
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                # Add record with empty pred on error and write immediately
                record_copy = record.copy()
                record_copy['pred'] = []
                output_file.write(json.dumps(record_copy, ensure_ascii=False) + '\n')
                output_file.flush()  # Ensure data is written to disk immediately
                processed_count += 1
    finally:
        output_file.close()
    
    print(f"\n{'=' * 80}")
    print(f"✓ Completed! Processed {processed_count} records")
    print(f"✓ Output saved to: {output_jsonl_path}")
    return output_jsonl_path


if __name__ == "__main__":
    import sys
    
    # Check if batch mode
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        # Batch mode: process jsonl file
        input_path = "/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/chensenda/codes/Neusoft/TestCaseGen/original_data/ai4test_neusoft_eval_data_30.jsonl"
        output_path = "/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/chensenda/codes/Neusoft/TestCaseGen/original_data/ai4test_neusoft_eval_data_30_with_pred.jsonl"
        
        batch_test_jsonl(
            input_jsonl_path=input_path,
            output_jsonl_path=output_path,
            backend="vLLM (Streaming)",
            use_rag=True,
            rag_topk=3,
            jsonl_path="/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/chensenda/codes/Neusoft/TestCaseGen/update_jsonl",
            additional_req=""
        )
    else:
        # Single test mode
        test_ui_automation_generation()