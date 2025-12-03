"""Data export utilities for test case generation"""
import os
import json
from datetime import datetime

# Export directory
EXPORT_DIR = os.path.join(os.path.dirname(__file__), "exported_data")


def ensure_export_dir():
    """Ensure export directory exists"""
    os.makedirs(EXPORT_DIR, exist_ok=True)


def export_data_to_dict(global_data):
    """Convert global_data to exportable dict"""
    export_data = {
        "export_time": datetime.now().isoformat(),
        "document": {
            "id": global_data.get("document_id"),
            "display_name": global_data.get("document_display_name")
        },
        "prd_text": global_data.get("prd_text", ""),
        "features": global_data.get("features", []),
        "test_points": {},
        "test_cases": {}
    }

    # Convert test_points keys to string
    for key, value in global_data.get("test_points", {}).items():
        export_data["test_points"][str(key)] = value

    # Convert test_cases tuple keys to string
    for key, value in global_data.get("test_cases", {}).items():
        if isinstance(key, tuple):
            str_key = f"{key[0]},{key[1]}"
        else:
            str_key = str(key)
        export_data["test_cases"][str_key] = value

    return export_data


def export_to_json_string(global_data):
    """Export data to JSON string"""
    export_data = export_data_to_dict(global_data)
    return json.dumps(export_data, ensure_ascii=False, indent=2)


def save_to_server(global_data, filename=None):
    """Save exported data to server and return file path"""
    ensure_export_dir()
    
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_name = global_data.get("document_display_name", "unknown")
        # Clean filename
        doc_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in doc_name)
        filename = f"export_{doc_name}_{timestamp}.json"
    
    filepath = os.path.join(EXPORT_DIR, filename)
    export_data = export_data_to_dict(global_data)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    return filepath


def get_saved_exports():
    """Get list of saved export files"""
    ensure_export_dir()
    files = []
    for f in os.listdir(EXPORT_DIR):
        if f.endswith(".json"):
            filepath = os.path.join(EXPORT_DIR, f)
            stat = os.stat(filepath)
            files.append({
                "name": f,
                "path": filepath,
                "size": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
    # Sort by modification time, newest first
    files.sort(key=lambda x: x["mtime"], reverse=True)
    return files


def load_export_file(filepath):
    """Load an exported JSON file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def format_data_for_visualization(global_data):
    """Format data for visualization in Markdown"""
    output = "# 📊 Test Case Data Visualization\n\n"
    
    # Document info
    doc_name = global_data.get("document_display_name", "N/A")
    output += f"## 📄 Document: {doc_name}\n\n"
    
    # Statistics
    features = global_data.get("features", [])
    test_points = global_data.get("test_points", {})
    test_cases = global_data.get("test_cases", {})
    
    total_test_points = sum(len(tp) for tp in test_points.values())
    total_test_cases = sum(len(tc) for tc in test_cases.values())
    
    output += "### 📈 Statistics\n\n"
    output += f"| Metric | Count |\n"
    output += f"|--------|-------|\n"
    output += f"| Features | {len(features)} |\n"
    output += f"| Test Points | {total_test_points} |\n"
    output += f"| Test Cases | {total_test_cases} |\n\n"
    
    # Features hierarchy
    output += "---\n\n## 🗂️ Features & Test Points\n\n"
    
    for i, feature in enumerate(features):
        feature_id = feature.get("id", i + 1)
        feature_name = feature.get("name", "Unknown")
        feature_desc = feature.get("description", "")
        
        output += f"### {feature_id}. {feature_name}\n"
        output += f"> {feature_desc}\n\n"
        
        # Test points for this feature
        feature_idx = i
        if feature_idx in test_points:
            tps = test_points[feature_idx]
            output += f"**Test Points ({len(tps)}):**\n\n"
            
            for tp in tps:
                tp_id = tp.get("id", "?")
                tp_name = tp.get("name", "Unknown")
                tp_type = tp.get("type", "N/A")
                tp_priority = tp.get("priority", "N/A")
                
                output += f"- **{tp_id}. {tp_name}**\n"
                output += f"  - Type: `{tp_type}` | Priority: `{tp_priority}`\n"
                
                # Test cases for this test point
                tc_key = (feature_idx, tp.get("id", 1) - 1)
                if tc_key in test_cases:
                    tcs = test_cases[tc_key]
                    output += f"  - Test Cases: {len(tcs)}\n"
            
            output += "\n"
        else:
            output += "*No test points generated*\n\n"
    
    # Detailed test cases section
    if test_cases:
        output += "---\n\n## 📝 Test Cases Detail\n\n"
        
        for key, cases in test_cases.items():
            if isinstance(key, tuple):
                f_idx, tp_idx = key
            else:
                parts = str(key).split(",")
                f_idx, tp_idx = int(parts[0]), int(parts[1])
            
            if f_idx < len(features):
                feature_name = features[f_idx].get("name", "Unknown")
            else:
                feature_name = "Unknown"
            
            if f_idx in test_points and tp_idx < len(test_points[f_idx]):
                tp_name = test_points[f_idx][tp_idx].get("name", "Unknown")
            else:
                tp_name = "Unknown"
            
            output += f"### Feature: {feature_name} > Test Point: {tp_name}\n\n"
            
            for tc in cases:
                case_id = tc.get("case_id", "?")
                title = tc.get("title", "Unknown")
                priority = tc.get("priority", "N/A")
                
                output += f"#### {case_id}: {title}\n"
                output += f"- **Priority**: {priority}\n"
                output += f"- **Precondition**: {tc.get('precondition', 'N/A')}\n"
                output += f"- **Expected Result**: {tc.get('expected_result', 'N/A')}\n"
                
                steps = tc.get("test_steps", [])
                if steps:
                    output += "- **Steps**:\n"
                    for step in steps:
                        output += f"  {step.get('step', '?')}. {step.get('action', 'N/A')}\n"
                
                output += "\n"
    
    return output


def format_data_for_labeling(global_data):
    """Format data for labeling in a structured table view"""
    output = "# 🏷️ Data Labeling View\n\n"
    output += "Use this view to review and label test cases.\n\n"
    
    features = global_data.get("features", [])
    test_points = global_data.get("test_points", {})
    test_cases = global_data.get("test_cases", {})
    
    # Create a flat table of all test cases
    output += "## Test Cases Table\n\n"
    output += "| Feature | Test Point | Case ID | Title | Priority | Status |\n"
    output += "|---------|------------|---------|-------|----------|--------|\n"
    
    for key, cases in test_cases.items():
        if isinstance(key, tuple):
            f_idx, tp_idx = key
        else:
            parts = str(key).split(",")
            f_idx, tp_idx = int(parts[0]), int(parts[1])
        
        feature_name = features[f_idx].get("name", "?") if f_idx < len(features) else "?"
        
        if f_idx in test_points and tp_idx < len(test_points[f_idx]):
            tp_name = test_points[f_idx][tp_idx].get("name", "?")
        else:
            tp_name = "?"
        
        for tc in cases:
            case_id = tc.get("case_id", "?")
            title = tc.get("title", "?")[:30] + "..." if len(tc.get("title", "")) > 30 else tc.get("title", "?")
            priority = tc.get("priority", "?")
            
            output += f"| {feature_name[:20]} | {tp_name[:20]} | {case_id} | {title} | {priority} | ⬜ |\n"
    
    return output

