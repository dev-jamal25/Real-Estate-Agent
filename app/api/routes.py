"""API routes for Stage 1 extraction, Stage 2 prediction, and Stage 3 interpretation."""
import logging

from fastapi import APIRouter, Depends
from pydantic import ValidationError

from app.api.dependencies import (
    get_extraction_service,
    get_interpretation_service,
    get_prediction_service,
)
from app.api.schemas import HealthResponse, PredictRequest, PredictResponse
from app.extraction.service import ExtractionService
from app.interpretation.service import InterpretationService
from app.prediction.schemas import CompletePropertyFeatures
from app.prediction.service import PredictionService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint.
    
    Returns:
        HealthResponse with status='healthy'
    """
    return HealthResponse(status="healthy")


@router.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    extraction_service: ExtractionService = Depends(get_extraction_service),
    prediction_service: PredictionService = Depends(get_prediction_service),
    interpretation_service: InterpretationService = Depends(get_interpretation_service),
) -> PredictResponse:
    
    logger.info(f"Predict request: {len(request.query)} chars, {len(request.accumulated_features)} accumulated")

    # Stage 1: Extract features
    extraction_response = extraction_service.extract(
        latest_message=request.query,
        accumulated_state=request.accumulated_features,
    )

    # Merge: previous accumulated + latest extracted
    previous_state = request.accumulated_features
    latest_extracted = extraction_response.extracted_features.model_dump(exclude_none=True)
    merged_dict = {**previous_state, **latest_extracted}

    logger.info(
        f"Extraction complete: merged {len(merged_dict)}, new {len(latest_extracted)}, missing {len(extraction_response.missing_features)}"
    )

    # Initialize response with extraction results
    response_data = {
        "accumulated_features": merged_dict,
        "extracted_features": latest_extracted,
        "missing_features": extraction_response.missing_features,
        "is_complete": extraction_response.is_complete,
        "reply": extraction_response.reply,
        "predicted_price": None,
        "global_stats": {},
        "interpretation": None,
    }

    # Stage 2: Predict only if complete
    if extraction_response.is_complete:
        try:
            # Validate complete feature set for prediction
            complete_features = CompletePropertyFeatures(**merged_dict)
            logger.debug(f"Complete features validated for prediction")
            
            # Run prediction
            prediction_output = prediction_service.predict(complete_features)
            
            response_data["predicted_price"] = prediction_output.predicted_price
            response_data["global_stats"] = prediction_output.global_stats
            
            logger.info(
                f"Prediction complete: ${prediction_output.predicted_price:.2f}"
            )
            
            # Stage 3: Interpret only if prediction succeeded
            try:
                # Use validated complete_features model (dumped to dict)
                complete_features_dict = complete_features.model_dump()
                interpretation = interpretation_service.interpret(
                    complete_features=complete_features_dict,
                    predicted_price=prediction_output.predicted_price,
                    global_stats=prediction_output.global_stats,
                )
                response_data["interpretation"] = interpretation
                logger.info(f"Interpretation complete: {len(interpretation)} chars")
            except Exception as e:
                logger.error(f"Interpretation failed: {e}")
                # Return prediction without interpretation (graceful degradation)
            
        except ValidationError as e:
            logger.error(f"Complete feature validation failed: {e}")
            # Return extraction response without prediction
            # (don't fail the API call, just skip prediction)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return extraction response without prediction
            # (graceful degradation)

    # Build the API response
    return PredictResponse(**response_data)
