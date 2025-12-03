"""
Quality evaluation module for generated features, test points, and test cases.

Provides scoring functions to assess the quality of generated content.
"""
from typing import List, Dict, Tuple


def evaluate_features(features: List[dict], prd_text: str = "") -> Dict:
    """
    Evaluate the quality of generated features.
    
    Scoring criteria:
    - Completeness: Each feature has id, name, description
    - Clarity: Description length and detail
    - Quantity: Reasonable number of features
    - Uniqueness: No duplicate names
    
    Returns:
        {
            "total_score": float (0-100),
            "details": {
                "completeness": {"score": float, "max": 30, "issues": []},
                "clarity": {"score": float, "max": 30, "issues": []},
                "quantity": {"score": float, "max": 20, "issues": []},
                "uniqueness": {"score": float, "max": 20, "issues": []}
            },
            "summary": str
        }
    """
    if not features:
        return {
            "total_score": 0,
            "details": {},
            "summary": "No features to evaluate"
        }
    
    details = {}
    
    # 1. Completeness (30 points)
    completeness_score = 30
    completeness_issues = []
    for i, f in enumerate(features):
        missing = []
        if not f.get("id"):
            missing.append("id")
        if not f.get("name"):
            missing.append("name")
        if not f.get("description"):
            missing.append("description")
        if missing:
            completeness_issues.append(f"Feature {i+1}: missing {', '.join(missing)}")
            completeness_score -= 10 / len(features)
    
    details["completeness"] = {
        "score": max(0, completeness_score),
        "max": 30,
        "issues": completeness_issues
    }
    
    # 2. Clarity (30 points)
    clarity_score = 30
    clarity_issues = []
    for i, f in enumerate(features):
        desc = f.get("description", "")
        name = f.get("name", "")
        
        # Check description length
        if len(desc) < 20:
            clarity_issues.append(f"Feature {i+1}: description too short ({len(desc)} chars)")
            clarity_score -= 5
        elif len(desc) < 50:
            clarity_score -= 2
        
        # Check name length
        if len(name) < 3:
            clarity_issues.append(f"Feature {i+1}: name too short")
            clarity_score -= 3
        elif len(name) > 50:
            clarity_issues.append(f"Feature {i+1}: name too long")
            clarity_score -= 2
    
    details["clarity"] = {
        "score": max(0, clarity_score),
        "max": 30,
        "issues": clarity_issues
    }
    
    # 3. Quantity (20 points)
    quantity_score = 20
    quantity_issues = []
    num_features = len(features)
    
    if num_features < 2:
        quantity_issues.append(f"Too few features ({num_features})")
        quantity_score -= 10
    elif num_features < 3:
        quantity_score -= 5
    elif num_features > 20:
        quantity_issues.append(f"Too many features ({num_features}), may be too granular")
        quantity_score -= 5
    
    details["quantity"] = {
        "score": max(0, quantity_score),
        "max": 20,
        "issues": quantity_issues
    }
    
    # 4. Uniqueness (20 points)
    uniqueness_score = 20
    uniqueness_issues = []
    names = [f.get("name", "").lower().strip() for f in features]
    seen = set()
    duplicates = []
    for name in names:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    
    if duplicates:
        uniqueness_issues.append(f"Duplicate names: {', '.join(set(duplicates))}")
        uniqueness_score -= 10 * len(set(duplicates))
    
    details["uniqueness"] = {
        "score": max(0, uniqueness_score),
        "max": 20,
        "issues": uniqueness_issues
    }
    
    # Calculate total
    total_score = sum(d["score"] for d in details.values())
    
    # Generate summary
    if total_score >= 90:
        summary = "Excellent quality"
    elif total_score >= 75:
        summary = "Good quality"
    elif total_score >= 60:
        summary = "Acceptable quality"
    elif total_score >= 40:
        summary = "Needs improvement"
    else:
        summary = "Poor quality"
    
    return {
        "total_score": round(total_score, 1),
        "details": details,
        "summary": summary
    }


def evaluate_test_points(test_points: List[dict], feature: dict = None) -> Dict:
    """
    Evaluate the quality of generated test points.
    
    Scoring criteria:
    - Completeness: Required fields present
    - Coverage: Different test types covered
    - Priority: Proper priority distribution
    - Detail: Preconditions and expected results
    """
    if not test_points:
        return {
            "total_score": 0,
            "details": {},
            "summary": "No test points to evaluate"
        }
    
    details = {}
    
    # 1. Completeness (25 points)
    completeness_score = 25
    completeness_issues = []
    required_fields = ["id", "name", "type", "priority", "description"]
    
    for i, tp in enumerate(test_points):
        missing = [f for f in required_fields if not tp.get(f)]
        if missing:
            completeness_issues.append(f"Test point {i+1}: missing {', '.join(missing)}")
            completeness_score -= 5 / len(test_points)
    
    details["completeness"] = {
        "score": max(0, completeness_score),
        "max": 25,
        "issues": completeness_issues
    }
    
    # 2. Coverage (25 points)
    coverage_score = 25
    coverage_issues = []
    types = set(tp.get("type", "").lower() for tp in test_points)
    
    expected_types = {"functional", "performance", "security", "compatibility", "usability"}
    covered = types & expected_types
    
    if len(covered) < 2:
        coverage_issues.append(f"Limited test type coverage: only {types}")
        coverage_score -= 15
    elif len(covered) < 3:
        coverage_score -= 5
    
    # Check for positive/negative scenarios
    has_positive = any("positive" in tp.get("name", "").lower() or 
                       "normal" in tp.get("name", "").lower() or
                       "success" in tp.get("name", "").lower() 
                       for tp in test_points)
    has_negative = any("negative" in tp.get("name", "").lower() or 
                       "error" in tp.get("name", "").lower() or
                       "fail" in tp.get("name", "").lower() or
                       "invalid" in tp.get("name", "").lower()
                       for tp in test_points)
    
    if not has_negative:
        coverage_issues.append("No negative/error test scenarios found")
        coverage_score -= 5
    
    details["coverage"] = {
        "score": max(0, coverage_score),
        "max": 25,
        "issues": coverage_issues
    }
    
    # 3. Priority distribution (25 points)
    priority_score = 25
    priority_issues = []
    priorities = [tp.get("priority", "").lower() for tp in test_points]
    
    high_count = sum(1 for p in priorities if "high" in p or "高" in p)
    medium_count = sum(1 for p in priorities if "medium" in p or "中" in p)
    low_count = sum(1 for p in priorities if "low" in p or "低" in p)
    
    total = len(test_points)
    if high_count == total:
        priority_issues.append("All test points marked as high priority")
        priority_score -= 10
    elif high_count == 0:
        priority_issues.append("No high priority test points")
        priority_score -= 5
    
    details["priority"] = {
        "score": max(0, priority_score),
        "max": 25,
        "issues": priority_issues
    }
    
    # 4. Detail (25 points)
    detail_score = 25
    detail_issues = []
    
    for i, tp in enumerate(test_points):
        desc = tp.get("description", "")
        precond = tp.get("precondition", "")
        expected = tp.get("expected_result", "")
        
        if len(desc) < 20:
            detail_issues.append(f"Test point {i+1}: description too brief")
            detail_score -= 3
        if not precond or precond.lower() in ["none", "n/a", "无", ""]:
            detail_score -= 1
        if not expected or expected.lower() in ["none", "n/a", "无", ""]:
            detail_issues.append(f"Test point {i+1}: missing expected result")
            detail_score -= 3
    
    details["detail"] = {
        "score": max(0, detail_score),
        "max": 25,
        "issues": detail_issues
    }
    
    total_score = sum(d["score"] for d in details.values())
    
    if total_score >= 90:
        summary = "Excellent quality"
    elif total_score >= 75:
        summary = "Good quality"
    elif total_score >= 60:
        summary = "Acceptable quality"
    elif total_score >= 40:
        summary = "Needs improvement"
    else:
        summary = "Poor quality"
    
    return {
        "total_score": round(total_score, 1),
        "details": details,
        "summary": summary
    }


def evaluate_test_cases(test_cases: List[dict], test_point: dict = None) -> Dict:
    """
    Evaluate the quality of generated test cases.
    
    Scoring criteria:
    - Completeness: Required fields present
    - Steps: Test steps quality
    - Data: Test data provided
    - Traceability: Links to test point
    """
    if not test_cases:
        return {
            "total_score": 0,
            "details": {},
            "summary": "No test cases to evaluate"
        }
    
    details = {}
    
    # 1. Completeness (25 points)
    completeness_score = 25
    completeness_issues = []
    required_fields = ["case_id", "title", "priority", "precondition", "expected_result"]
    
    for i, tc in enumerate(test_cases):
        missing = [f for f in required_fields if not tc.get(f)]
        if missing:
            completeness_issues.append(f"Test case {i+1}: missing {', '.join(missing)}")
            completeness_score -= 5 / len(test_cases)
    
    details["completeness"] = {
        "score": max(0, completeness_score),
        "max": 25,
        "issues": completeness_issues
    }
    
    # 2. Steps quality (30 points)
    steps_score = 30
    steps_issues = []
    
    for i, tc in enumerate(test_cases):
        steps = tc.get("test_steps", [])
        
        if not steps:
            steps_issues.append(f"Test case {i+1}: no test steps")
            steps_score -= 10
        elif len(steps) < 2:
            steps_issues.append(f"Test case {i+1}: too few steps ({len(steps)})")
            steps_score -= 5
        else:
            # Check step quality
            for j, step in enumerate(steps):
                if not step.get("action"):
                    steps_score -= 2
                if not step.get("expected"):
                    steps_score -= 1
    
    details["steps"] = {
        "score": max(0, steps_score),
        "max": 30,
        "issues": steps_issues
    }
    
    # 3. Test data (20 points)
    data_score = 20
    data_issues = []
    
    for i, tc in enumerate(test_cases):
        test_data = tc.get("test_data", "")
        if not test_data or test_data.lower() in ["none", "n/a", "无", ""]:
            data_issues.append(f"Test case {i+1}: no test data provided")
            data_score -= 5
    
    details["test_data"] = {
        "score": max(0, data_score),
        "max": 20,
        "issues": data_issues
    }
    
    # 4. Clarity (25 points)
    clarity_score = 25
    clarity_issues = []
    
    for i, tc in enumerate(test_cases):
        title = tc.get("title", "")
        expected = tc.get("expected_result", "")
        
        if len(title) < 10:
            clarity_issues.append(f"Test case {i+1}: title too short")
            clarity_score -= 3
        if len(expected) < 10:
            clarity_issues.append(f"Test case {i+1}: expected result too brief")
            clarity_score -= 3
    
    # Check for unique case IDs
    case_ids = [tc.get("case_id", "") for tc in test_cases]
    if len(case_ids) != len(set(case_ids)):
        clarity_issues.append("Duplicate case IDs found")
        clarity_score -= 5
    
    details["clarity"] = {
        "score": max(0, clarity_score),
        "max": 25,
        "issues": clarity_issues
    }
    
    total_score = sum(d["score"] for d in details.values())
    
    if total_score >= 90:
        summary = "Excellent quality"
    elif total_score >= 75:
        summary = "Good quality"
    elif total_score >= 60:
        summary = "Acceptable quality"
    elif total_score >= 40:
        summary = "Needs improvement"
    else:
        summary = "Poor quality"
    
    return {
        "total_score": round(total_score, 1),
        "details": details,
        "summary": summary
    }


def format_evaluation_markdown(eval_result: Dict, title: str = "Quality Evaluation") -> str:
    """Format evaluation result as Markdown for display"""
    output = f"## 📊 {title}\n\n"
    
    total = eval_result.get("total_score", 0)
    summary = eval_result.get("summary", "N/A")
    
    # Score badge
    if total >= 90:
        badge = "🟢"
    elif total >= 75:
        badge = "🟡"
    elif total >= 60:
        badge = "🟠"
    else:
        badge = "🔴"
    
    output += f"**Overall Score: {badge} {total}/100** ({summary})\n\n"
    
    # Details table
    output += "| Criterion | Score | Max | Status |\n"
    output += "|-----------|-------|-----|--------|\n"
    
    for name, detail in eval_result.get("details", {}).items():
        score = detail.get("score", 0)
        max_score = detail.get("max", 0)
        pct = (score / max_score * 100) if max_score > 0 else 0
        
        if pct >= 90:
            status = "✅"
        elif pct >= 70:
            status = "⚠️"
        else:
            status = "❌"
        
        output += f"| {name.title()} | {score:.1f} | {max_score} | {status} |\n"
    
    # Issues
    all_issues = []
    for detail in eval_result.get("details", {}).values():
        all_issues.extend(detail.get("issues", []))
    
    if all_issues:
        output += "\n### ⚠️ Issues Found\n\n"
        for issue in all_issues[:10]:  # Limit to 10 issues
            output += f"- {issue}\n"
        if len(all_issues) > 10:
            output += f"- ... and {len(all_issues) - 10} more issues\n"
    
    return output

