"""
FastAPI service for test case generation core functions.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8080 --reload

API Endpoints:
    POST /api/features       - Extract features from PRD
    POST /api/test-points    - Generate test points for a feature
    POST /api/test-cases     - Generate test cases for a test point
    POST /api/full-pipeline  - Run full pipeline (PRD -> features -> test points -> test cases)
"""
import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_util import create_vllm_client, generate_response_vllm
from parse_util import robust_json_parse
from prompt_util import load_prompt_template

# Initialize FastAPI app
app = FastAPI(
    title="Test Case Generation API",
    description="API for extracting features, generating test points and test cases from PRD documents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global vLLM client
vllm_client = None
model_id = None


# ========== Request/Response Models ==========

class ModelConfig(BaseModel):
    base_url: str = "http://localhost:12349/v1"
    model_id: str = "Qwen3-8B"


class FeaturesRequest(BaseModel):
    prd_text: str
    additional_requirement: Optional[str] = ""


class FeaturesResponse(BaseModel):
    success: bool
    features: List[dict]
    raw_response: Optional[str] = None
    error: Optional[str] = None


class TestPointsRequest(BaseModel):
    prd_text: str
    feature: dict  # {"id": 1, "name": "...", "description": "..."}
    additional_requirement: Optional[str] = ""


class TestPointsResponse(BaseModel):
    success: bool
    test_points: List[dict]
    raw_response: Optional[str] = None
    error: Optional[str] = None


class TestCasesRequest(BaseModel):
    prd_text: str
    feature: dict
    test_point: dict
    additional_requirement: Optional[str] = ""


class TestCasesResponse(BaseModel):
    success: bool
    test_cases: List[dict]
    raw_response: Optional[str] = None
    error: Optional[str] = None


class FullPipelineRequest(BaseModel):
    prd_text: str
    additional_requirement: Optional[str] = ""
    max_features: Optional[int] = None  # Limit features to process
    max_test_points_per_feature: Optional[int] = None


class FullPipelineResponse(BaseModel):
    success: bool
    features: List[dict]
    test_points: dict  # {feature_id: [test_points]}
    test_cases: dict  # {"feature_id,test_point_id": [test_cases]}
    error: Optional[str] = None


# ========== API Endpoints ==========

@app.post("/api/init", summary="Initialize vLLM client")
async def init_model(config: ModelConfig):
    """Initialize the vLLM client with given configuration"""
    global vllm_client, model_id
    try:
        vllm_client = create_vllm_client(config.base_url)
        model_id = config.model_id
        return {"success": True, "message": f"Initialized with {config.base_url}, model: {config.model_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status", summary="Check API status")
async def get_status():
    """Check if the API and model are ready"""
    return {
        "status": "ready" if vllm_client else "not_initialized",
        "model_id": model_id
    }


@app.post("/api/features", response_model=FeaturesResponse, summary="Extract features from PRD")
async def extract_features(request: FeaturesRequest):
    """Extract features from PRD document"""
    if not vllm_client:
        raise HTTPException(status_code=400, detail="Model not initialized. Call /api/init first.")
    
    try:
        system_prompt = load_prompt_template("generate_features_system_prompt")
        user_prompt_template = load_prompt_template("generate_features_user_prompt")
        user_prompt = user_prompt_template.format(prd_text=request.prd_text)
        
        if request.additional_requirement:
            user_prompt += f"\n\nAdditional requirement: {request.additional_requirement}"
        
        response = generate_response_vllm(vllm_client, model_id, user_prompt, system_prompt)
        result = robust_json_parse(response)
        
        features = result.get("features", [])
        return FeaturesResponse(success=True, features=features, raw_response=response)
    
    except Exception as e:
        return FeaturesResponse(success=False, features=[], error=str(e))


@app.post("/api/test-points", response_model=TestPointsResponse, summary="Generate test points")
async def generate_test_points(request: TestPointsRequest):
    """Generate test points for a given feature"""
    if not vllm_client:
        raise HTTPException(status_code=400, detail="Model not initialized. Call /api/init first.")
    
    try:
        system_prompt = load_prompt_template("generate_test_points_system_prompt")
        user_prompt_template = load_prompt_template("generate_test_points_user_prompt")
        user_prompt = user_prompt_template.format(
            feature_name=request.feature.get("name", ""),
            feature_description=request.feature.get("description", ""),
            prd_text=request.prd_text
        )
        
        if request.additional_requirement:
            user_prompt += f"\n\nAdditional requirement: {request.additional_requirement}"
        
        response = generate_response_vllm(vllm_client, model_id, user_prompt, system_prompt)
        result = robust_json_parse(response)
        
        test_points = result.get("test_points", [])
        return TestPointsResponse(success=True, test_points=test_points, raw_response=response)
    
    except Exception as e:
        return TestPointsResponse(success=False, test_points=[], error=str(e))


@app.post("/api/test-cases", response_model=TestCasesResponse, summary="Generate test cases")
async def generate_test_cases(request: TestCasesRequest):
    """Generate test cases for a given test point"""
    if not vllm_client:
        raise HTTPException(status_code=400, detail="Model not initialized. Call /api/init first.")
    
    try:
        system_prompt = load_prompt_template("generate_test_cases_system_prompt")
        user_prompt_template = load_prompt_template("generate_test_cases_user_prompt")
        user_prompt = user_prompt_template.format(
            feature_name=request.feature.get("name", ""),
            test_point_name=request.test_point.get("name", ""),
            test_point_description=request.test_point.get("description", ""),
            test_point_type=request.test_point.get("type", ""),
            test_point_priority=request.test_point.get("priority", ""),
            test_point_precondition=request.test_point.get("precondition", ""),
            test_point_expected_result=request.test_point.get("expected_result", ""),
            prd_text=request.prd_text
        )
        
        if request.additional_requirement:
            user_prompt += f"\n\nAdditional requirement: {request.additional_requirement}"
        
        response = generate_response_vllm(vllm_client, model_id, user_prompt, system_prompt)
        result = robust_json_parse(response)
        
        test_cases = result.get("test_cases", [])
        return TestCasesResponse(success=True, test_cases=test_cases, raw_response=response)
    
    except Exception as e:
        return TestCasesResponse(success=False, test_cases=[], error=str(e))


@app.post("/api/full-pipeline", response_model=FullPipelineResponse, summary="Run full pipeline")
async def run_full_pipeline(request: FullPipelineRequest):
    """
    Run the full pipeline: PRD -> Features -> Test Points -> Test Cases
    
    This endpoint processes the entire workflow in one call.
    Use max_features and max_test_points_per_feature to limit processing scope.
    """
    if not vllm_client:
        raise HTTPException(status_code=400, detail="Model not initialized. Call /api/init first.")
    
    try:
        # Step 1: Extract features
        features_response = await extract_features(FeaturesRequest(
            prd_text=request.prd_text,
            additional_requirement=request.additional_requirement
        ))
        
        if not features_response.success:
            return FullPipelineResponse(
                success=False, features=[], test_points={}, test_cases={},
                error=f"Failed to extract features: {features_response.error}"
            )
        
        features = features_response.features
        if request.max_features:
            features = features[:request.max_features]
        
        # Step 2: Generate test points for each feature
        all_test_points = {}
        for i, feature in enumerate(features):
            tp_response = await generate_test_points(TestPointsRequest(
                prd_text=request.prd_text,
                feature=feature,
                additional_requirement=request.additional_requirement
            ))
            
            if tp_response.success:
                test_points = tp_response.test_points
                if request.max_test_points_per_feature:
                    test_points = test_points[:request.max_test_points_per_feature]
                all_test_points[str(i)] = test_points
        
        # Step 3: Generate test cases for each test point
        all_test_cases = {}
        for feature_idx, test_points in all_test_points.items():
            feature = features[int(feature_idx)]
            for tp_idx, test_point in enumerate(test_points):
                tc_response = await generate_test_cases(TestCasesRequest(
                    prd_text=request.prd_text,
                    feature=feature,
                    test_point=test_point,
                    additional_requirement=request.additional_requirement
                ))
                
                if tc_response.success:
                    key = f"{feature_idx},{tp_idx}"
                    all_test_cases[key] = tc_response.test_cases
        
        return FullPipelineResponse(
            success=True,
            features=features,
            test_points=all_test_points,
            test_cases=all_test_cases
        )
    
    except Exception as e:
        return FullPipelineResponse(
            success=False, features=[], test_points={}, test_cases={},
            error=str(e)
        )


# ========== Main ==========

if __name__ == "__main__":
    import uvicorn
    
    # Set proxy bypass for localhost
    no_proxy = os.environ.get('NO_PROXY', os.environ.get('no_proxy', ''))
    if no_proxy:
        os.environ['NO_PROXY'] = no_proxy + ',localhost,127.0.0.1,0.0.0.0'
        os.environ['no_proxy'] = no_proxy + ',localhost,127.0.0.1,0.0.0.0'
    else:
        os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'
        os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'
    
    uvicorn.run(app, host="0.0.0.0", port=8080)

