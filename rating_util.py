"""
Manual rating utilities for generated features, test points, and test cases.

Allows users to manually score and annotate generated content.
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional

# Rating storage directory
RATING_DIR = os.path.join(os.path.dirname(__file__), "ratings")


def ensure_rating_dir():
    """Ensure rating directory exists"""
    os.makedirs(RATING_DIR, exist_ok=True)


def save_rating(rating_data: Dict) -> str:
    """
    Save a rating record to file.
    
    rating_data should contain:
    - type: "features" | "test_points" | "test_cases"
    - document_id: str
    - item_id: str (feature id, test point id, or test case id)
    - score: int (1-5)
    - comment: str
    - timestamp: str (auto-generated if not provided)
    """
    ensure_rating_dir()
    
    if "timestamp" not in rating_data:
        rating_data["timestamp"] = datetime.now().isoformat()
    
    # Load existing ratings
    ratings_file = os.path.join(RATING_DIR, "ratings.json")
    if os.path.exists(ratings_file):
        with open(ratings_file, "r", encoding="utf-8") as f:
            ratings = json.load(f)
    else:
        ratings = []
    
    # Add new rating
    ratings.append(rating_data)
    
    # Save back
    with open(ratings_file, "w", encoding="utf-8") as f:
        json.dump(ratings, f, ensure_ascii=False, indent=2)
    
    return rating_data["timestamp"]


def get_ratings(rating_type: Optional[str] = None, document_id: Optional[str] = None) -> List[Dict]:
    """Get all ratings, optionally filtered by type and document"""
    ratings_file = os.path.join(RATING_DIR, "ratings.json")
    if not os.path.exists(ratings_file):
        return []
    
    with open(ratings_file, "r", encoding="utf-8") as f:
        ratings = json.load(f)
    
    if rating_type:
        ratings = [r for r in ratings if r.get("type") == rating_type]
    if document_id:
        ratings = [r for r in ratings if r.get("document_id") == document_id]
    
    return ratings


def get_rating_summary(document_id: Optional[str] = None) -> Dict:
    """Get rating statistics summary"""
    ratings = get_ratings(document_id=document_id)
    
    if not ratings:
        return {
            "total_ratings": 0,
            "features": {"count": 0, "avg_score": 0},
            "test_points": {"count": 0, "avg_score": 0},
            "test_cases": {"count": 0, "avg_score": 0}
        }
    
    summary = {
        "total_ratings": len(ratings),
        "features": {"count": 0, "scores": []},
        "test_points": {"count": 0, "scores": []},
        "test_cases": {"count": 0, "scores": []}
    }
    
    for r in ratings:
        rtype = r.get("type", "")
        score = r.get("score", 0)
        if rtype in summary:
            summary[rtype]["count"] += 1
            summary[rtype]["scores"].append(score)
    
    # Calculate averages
    for key in ["features", "test_points", "test_cases"]:
        scores = summary[key]["scores"]
        summary[key]["avg_score"] = round(sum(scores) / len(scores), 2) if scores else 0
        del summary[key]["scores"]
    
    return summary


def export_ratings_csv() -> str:
    """Export all ratings to CSV format"""
    ratings = get_ratings()
    
    if not ratings:
        return "No ratings to export"
    
    lines = ["timestamp,type,document_id,item_id,item_name,score,comment"]
    for r in ratings:
        line = f"{r.get('timestamp', '')},{r.get('type', '')},{r.get('document_id', '')},{r.get('item_id', '')},{r.get('item_name', '').replace(',', ';')},{r.get('score', '')},{r.get('comment', '').replace(',', ';')}"
        lines.append(line)
    
    return "\n".join(lines)


def format_rating_form_markdown(item_type: str, item_name: str, item_id: str) -> str:
    """Generate markdown description for rating form"""
    return f"""
### 📝 Rate: {item_name}

**Type:** {item_type}  
**ID:** {item_id}

Please provide your rating (1-5 stars) and any comments below.
"""

