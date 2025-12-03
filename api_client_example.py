"""
Example client for Test Case Generation API

Usage:
    1. Start the API server: python api.py
    2. Run this script: python api_client_example.py
"""
import requests

API_BASE = "http://localhost:8080"


def init_model(base_url="http://localhost:12349/v1", model_id="Qwen3-8B"):
    """Initialize the model"""
    response = requests.post(f"{API_BASE}/api/init", json={
        "base_url": base_url,
        "model_id": model_id
    })
    return response.json()


def extract_features(prd_text, additional_requirement=""):
    """Extract features from PRD"""
    response = requests.post(f"{API_BASE}/api/features", json={
        "prd_text": prd_text,
        "additional_requirement": additional_requirement
    })
    return response.json()


def generate_test_points(prd_text, feature, additional_requirement=""):
    """Generate test points for a feature"""
    response = requests.post(f"{API_BASE}/api/test-points", json={
        "prd_text": prd_text,
        "feature": feature,
        "additional_requirement": additional_requirement
    })
    return response.json()


def generate_test_cases(prd_text, feature, test_point, additional_requirement=""):
    """Generate test cases for a test point"""
    response = requests.post(f"{API_BASE}/api/test-cases", json={
        "prd_text": prd_text,
        "feature": feature,
        "test_point": test_point,
        "additional_requirement": additional_requirement
    })
    return response.json()


def run_full_pipeline(prd_text, additional_requirement="", max_features=None, max_test_points=None):
    """Run the full pipeline"""
    response = requests.post(f"{API_BASE}/api/full-pipeline", json={
        "prd_text": prd_text,
        "additional_requirement": additional_requirement,
        "max_features": max_features,
        "max_test_points_per_feature": max_test_points
    })
    return response.json()


if __name__ == "__main__":
    # Example PRD text
    prd_text = """
    # User Login Feature
    
    ## Overview
    Users can log in to the system using email and password.
    
    ## Requirements
    1. Users enter email and password on the login page
    2. System validates credentials
    3. On success, redirect to dashboard
    4. On failure, show error message
    5. Support "Remember me" option
    6. Support password reset via email
    """
    
    # Initialize model
    print("Initializing model...")
    result = init_model()
    print(f"Init result: {result}")
    
    # Extract features
    print("\nExtracting features...")
    features_result = extract_features(prd_text)
    print(f"Features: {features_result}")
    
    if features_result.get("success") and features_result.get("features"):
        feature = features_result["features"][0]
        
        # Generate test points
        print(f"\nGenerating test points for: {feature['name']}...")
        tp_result = generate_test_points(prd_text, feature)
        print(f"Test points: {tp_result}")
        
        if tp_result.get("success") and tp_result.get("test_points"):
            test_point = tp_result["test_points"][0]
            
            # Generate test cases
            print(f"\nGenerating test cases for: {test_point['name']}...")
            tc_result = generate_test_cases(prd_text, feature, test_point)
            print(f"Test cases: {tc_result}")
    
    # Or run full pipeline
    print("\n" + "="*50)
    print("Running full pipeline (limited to 1 feature, 2 test points)...")
    pipeline_result = run_full_pipeline(prd_text, max_features=1, max_test_points=2)
    print(f"Pipeline result: {pipeline_result}")

