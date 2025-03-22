import json
import os
import llmevents as llme

def load_av_brand_mappings():
    """
    Load the AV brand mappings from the configuration file.
    
    Returns:
        dict: Dictionary containing all brand mappings and related information
    """
    logger = llme.CustomLogger(__name__)
    
    # Direct path to the JSON file in the analysis directory
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis', 'av_brands.json')
    
    try:
        with open(json_path, 'r') as f:
            mappings = json.load(f)
        return mappings
    except Exception as e:
        logger.error(f"Failed to load AV brand mappings: {e}")
        # Return empty dictionaries as a fallback
        return {
            "brand_mapping": {},
            "av_company_mapping": {},
            "model_mapping": {},
            "specific_vehicle_patterns": {}
        } 