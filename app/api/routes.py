"""API routes: /predict (POST) and /health (GET)."""
import logging
from typing import Union

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_artifact, get_extractor, get_predictor, get_interpreter
from app.core.validators import compute_missing_features, is_complete
from app.models.feature_models import PropertyFeaturesComplete, PropertyFeaturesPartial
from app.models.request_models import PredictRequest
from app.models.response_models import (
    HealthResponse,
    IncompleteFeatureResponse,
    PredictionResponse,
)
from app.ml.loader import ModelArtifact
from app.ml.predictor import Predictor, PredictionError
from app.llm.extractor import Stage1Extractor
from app.llm.interpreter import Stage2Interpreter

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    artifact: ModelArtifact = Depends(get_artifact),
) -> HealthResponse:
    """
    Health check endpoint.
    
    Verifies that the service is running and the ML artifact is loaded.
    
    Returns:
        HealthResponse with status "ok" if healthy
    """
    try:
        # Verify artifact is accessible
        _ = artifact.model
        _ = artifact.features
        return HealthResponse(
            status="ok",
            message="Service healthy, model artifact loaded",
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy: artifact not available",
        ) from e


@router.post("/predict", response_model=Union[IncompleteFeatureResponse, PredictionResponse])
async def predict(
    request: PredictRequest,
    extractor: Stage1Extractor = Depends(get_extractor),
    predictor: Predictor = Depends(get_predictor),
    interpreter: Stage2Interpreter = Depends(get_interpreter),
) -> Union[IncompleteFeatureResponse, PredictionResponse]:
    """
    Main prediction endpoint.
    
    Flow:
    1. Run Stage 1 extraction on query
    2. Merge with feature_overrides (overrides are the accumulated known state)
    3. Check completeness
    4. If incomplete: return needs_more_info with missing fields
    5. If complete: validate, run prediction, run interpretation, return result
    
    Args:
        request: PredictRequest with query and optional feature_overrides
        extractor: Stage1Extractor service (injected)
        predictor: Predictor service (injected)
        interpreter: Stage2Interpreter service (injected)
    
    Returns:
        IncompleteFeatureResponse if features are incomplete
        PredictionResponse if prediction succeeds
    
    Raises:
        HTTPException: If extraction or prediction fails
    """
    try:
        # Ensure feature_overrides exists (defaults to empty)
        current_state = request.feature_overrides or PropertyFeaturesPartial()

        # Run Stage 1 extraction
        logger.info("Running Stage 1 extraction")
        extraction_result = extractor.extract(
            latest_user_message=request.query,
            current_known_feature_state=current_state,
        )

        # Merge: accumulated state + newly extracted features
        accumulated_dict = current_state.model_dump(exclude_none=True)
        extracted_dict = extraction_result.extracted_features.model_dump(exclude_none=True)
        merged_dict = {**accumulated_dict, **extracted_dict}

        # Check completeness using merged state
        missing = compute_missing_features(extracted_dict, accumulated_dict)
        complete = is_complete(extracted_dict, accumulated_dict)

        logger.debug(
            f"Extraction result: complete={complete}, missing={len(missing)}, "
            f"merged_fields={len(merged_dict)}"
        )

        # If incomplete, return needs_more_info
        if not complete:
            accumulated_features = PropertyFeaturesPartial(**merged_dict)
            return IncompleteFeatureResponse(
                status="needs_more_info",
                extracted_features=extraction_result.extracted_features,
                accumulated_features=accumulated_features,
                missing_features=missing,
                assistant_message=extraction_result.assistant_message,
            )

        # If complete, validate and predict
        try:
            # Validate complete merged state
            complete_features = PropertyFeaturesComplete(**merged_dict)
        except Exception as e:
            logger.error(f"Failed to validate complete feature set: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid feature values for prediction: {str(e)}",
            ) from e

        # Run prediction
        try:
            predicted_price = predictor.predict(complete_features)
        except PredictionError as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction service failed",
            ) from e

        # Run Stage 2 interpretation
        try:
            logger.info("Running Stage 2 interpretation")
            global_stats = predictor.get_global_stats()
            interpretation = interpreter.interpret(
                complete_features=complete_features,
                predicted_price=predicted_price,
                global_stats=global_stats,
            )
        except Exception as e:
            logger.error(f"Interpretation failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Interpretation service failed",
            ) from e

        # Build response with interpretation
        global_stats = predictor.get_global_stats()
        return PredictionResponse(
            status="success",
            accumulated_features=complete_features,
            predicted_price=predicted_price,
            global_stats=global_stats,
            interpretation=interpretation,
        )

    except HTTPException:
        # Re-raise HTTPException
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /predict: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e

