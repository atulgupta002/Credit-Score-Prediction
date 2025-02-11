import joblib
import os
import json
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def model_fn(model_dir):
    try:
        model = joblib.load(os.path.join(model_dir, 'model.joblib'))
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

def input_fn(request_body, request_content_type):
    logger.info(f"Received content type: {request_content_type}")
    
    if request_content_type == 'application/json':
        try:
            data = json.loads(request_body)['features']
            
            if not isinstance(data, list):
                raise ValueError("Input data must be a list")
                
            if len(data) != 20:  # Update with your actual feature count
                raise ValueError(f"Expected 20 features, got {len(data)}")
                
            return data
        except Exception as e:
            logger.error(f"Input parsing error: {str(e)}")
            raise
    else:
        error_msg = f"Unsupported content type: {request_content_type}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def predict_fn(input_data, model):
    try:
        logger.info("Making prediction")
        
        # For classification models
        if hasattr(model, "predict_proba"):
            prediction = model.predict([input_data])[0]
            probabilities = model.predict_proba([input_data])[0].tolist()
            return {
                'class': prediction,
                'probabilities': probabilities
            }
        # For regression models
        else:
            return float(model.predict([input_data])[0])
            
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

def output_fn(prediction, content_type):
    try:
        if content_type == 'application/json':
            response = {
                'result': prediction,
                'status': 'success',
                'model_type': 'classification' if isinstance(prediction, dict) else 'regression'
            }
            return json.dumps(response), content_type
        else:
            error_msg = f"Unsupported content type: {content_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    except Exception as e:
        error_response = {
            'error': str(e),
            'status': 'failed'
        }
        return json.dumps(error_response), 'application/json'