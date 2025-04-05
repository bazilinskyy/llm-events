# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com> and Linghan Zhang
import os
import pandas as pd
from tqdm import tqdm
import openai
from pdf2image import convert_from_path
import base64
from PIL import Image
import time
from openai import OpenAI
import re
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string

import llmevents as llme

# warning about partial assignment
pd.options.mode.chained_assignment = None  # default='warn'

logger = llme.CustomLogger(__name__)  # use custom logger

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Initialize LM Studio client
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
MODEL = "gemma-3-12b-it"


class LLMEvents:
    # pandas dataframe with extracted data
    data = pd.DataFrame()

    def __init__(self,
                 files_reports: list,
                 save_p: bool,
                 load_p: bool,
                 save_csv: bool):
        # list of files with raw data
        self.files_reports = files_reports
        # save data as pickle file
        self.save_p = save_p
        # load data as pickle file
        self.load_p = load_p
        # save data as csv file
        self.save_csv = save_csv
        # client for communicating with GPT4-V
        # self.gpt_client = openai.OpenAI(api_key=llme.common.get_secrets('openai_api_key'))
        self.gpt_client = client

    def read_data(self, filter_data=True, clean_data=True, analyse_data=True, save_interval=20, process_answers=True):
        """Read data into an attribute.

        Args:
            filter_data (bool, optional): flag for filtering data.
            clean_data (bool, optional): clean data.
            analyse_data (bool, optional): analyse data.
            save_interval (int, optional): save data after processing this many files.

        Returns:
            dataframe: updated dataframe.
        """
        # analyse
        if llme.common.get_configs('analyse'):
            # pandas df to store data
            df = pd.DataFrame(columns=('report', 'response'))
            # df = df.transpose()
            file_list = os.listdir(self.files_reports)
            # go over all reports
            for i, file in enumerate(tqdm(file_list)):
                logger.info('Processing report {}.', file)
                # get pages as base64_image strings
                pages = self.pdf_to_base64_image(file, resize_image=True)
                # feed all pages in the report to GPT-4V at once
                df = pd.concat([df, self.ask_llm(file, pages)], ignore_index=True)
                
                # Save periodically based on the interval
                if (i + 1) % save_interval == 0 or i == len(file_list) - 1:
                    logger.info('Periodic save after processing {} files.', i + 1)
                    if self.save_p:
                        llme.common.save_to_p(self.file_p, df, 'chat data (periodic)')
                    if self.save_csv:
                        periodic_csv = f"periodic_{i+1}_{llme.common.get_configs('data')}"
                        df.to_csv(os.path.join(llme.settings.output_dir, periodic_csv), index=False)
                        # Also save to the main file
                        df.to_csv(os.path.join(llme.settings.output_dir, 'data.csv'), index=False)
                        logger.info('Saved periodic data to csv file {}', periodic_csv)
            # report people that attempted study
            logger.info('Processed {} reports.', df.shape[0])
        # load from CSV instead
        else:
            df = pd.read_csv(llme.common.get_configs('data'))
        # clean data
        if clean_data:
            df = self.clean_data(df)
        # filter data
        if filter_data:
            df = self.filter_data(df)
        # process answers to the questions in the query
        if process_answers:
            df = self.process_answers(df)
        # # sort columns alphabetically
        # df = df.reindex(sorted(df.columns), axis=1)
        # save csv file with categorisation of answers
        if self.save_csv:
            df.to_csv(os.path.join(llme.settings.output_dir, 'data_processed.csv'), index=False)
        # return df with data
        return df

    def extract_answers(self, response, q):
        """Extract answers to each question from response text."""        
        answers = {}
        pattern = rf"(?:\*\*|)Q{q}[^\*]*?(?:\.|:)(.*?)(?=\n(?:\*\*|)Q{q+1}|$)"

        match = re.search(pattern, response, re.DOTALL)
        answers[f"q{q}"] = match.group(1).strip() if match else ""
        return answers

    def categorise(self, response, q, row_index):
        """Categorise responses to question 1.
        
        Args:
            response (str): response.
            q (str): question number.
            row_index (int): index of row for logging.
        
        Returns:
            str: categorisation.
        """        
        if q == "q1":
            if "Yes," in response or "Yes." in response:
                return "Yes"
            elif "No," in response or "No." in response:
                return "No"
            else:
                return "Other"
        
        elif q == "q2-av":
            brand_av = None
            model_av = None
            year_av = None
            # Cleanup of formatting
            response = re.sub(r"Autonomous Vehicle:|" +
                              r"Vehicle 1 \(Autonomous Vehicle\):|" +
                              r"Vehicle 1 \(Automated Vehicle\):|" +
                              r"The automated vehicle was a|" +
                              r"The autonomous vehicle was a|" +
                              r"One vehicle was involved: a|" +
                              r"Vehicle 1:", "Automated Vehicle:", response)
            response = re.sub(r"\*\*Automated Vehicle:\*\*", "Automated Vehicle:", response)
            response = re.sub(r"Apple Inc.", "Apple", response)
            response = re.sub(r"\(Not Specified\)", "Unknown", response)
            response = re.sub(r" \(indicated by a blank space\)", "", response)
            response = re.sub(r"Year:", "Year", response)
            response = re.sub(r"Brand:", "Brand", response)
            response = re.sub(r"Model:", "Model", response)
            response = re.sub(r"Unknown \(likely a Tesla\)", "Tesla", response)
            response = re.sub(r"Unknown \(likely Tesla based on the form\)", "Tesla", response)
            response = re.sub(r"Unknown \(likely Toyota based on the form\)", "Toyota", response)
            response = re.sub(r"Unknown \(indicated by \"AV\"\)", "Unknown", response)
            
            # Load brand mappings from configuration file
            from llmevents.utils import load_av_brand_mappings
            mappings = load_av_brand_mappings()
            
            # Check for specific vehicle patterns
            specific_vehicle_patterns = mappings.get("specific_vehicle_patterns", {})
            for pattern, info in specific_vehicle_patterns.items():
                if re.search(pattern, response, re.IGNORECASE):
                    brand_av = info["brand"]
                    model_av = info["model"] if info["model"] else model_av
                    year_av = info["year"] if info["year"] else year_av
                    break
            
            # If no specific pattern matched, try more general extraction patterns
            if not brand_av:
                # Various regex patterns to extract year, brand, model
                extraction_patterns = [
                    # Year, brand, model
                    r"Automated Vehicle: Year (Unknown|\d{4}), Brand ([A-Za-z-]+), Model ([A-Za-z0-9\s]+)",
                    # Year, brand, model
                    r"Automated Vehicle: (Unknown|\d{4}), Brand ([A-Za-z-]+), Model ([A-Za-z0-9\s]+)",
                    # Year, brand
                    r"Automated Vehicle: Year (Unknown|\d{4}), Brand ([A-Za-z\s]+)\.",
                    # Improved pattern for "Year Brand Model" format with more flexible capture
                    r"Auto(?:mated|nomous) Vehicle:(?:\s*\*\*)?\s*(\d{4})?\s*([A-Za-z][A-Za-z0-9\s\-]*?)\s+([A-Za-z0-9][A-Za-z0-9\s\-]+?)(?:\.|\,|\s+The|\s+was|\s+operating)",
                    # Specific pattern for Cruise AV and similar formats
                    r"Auto(?:mated|nomous) Vehicle:(?:\s*\*\*)?\s*(\d{4})?\s*([A-Za-z][A-Za-z0-9\s\-]+)\s+AV(?:\.|\,|\s+The|\s+was|\s+operating)",
                    # Fallback pattern with broader capture
                    r"Auto(?:mated|nomous) Vehicle:(?:\s*\*\*)?\s*(?:(\d{4})\s+)?([A-Za-z][A-Za-z0-9\s\-]+?)(?:\.|\,|\s+The|\s+was|\s+operating)",
                    # Other
                    r"Automated Vehicle:\s*(\d{4})?\s*([^.,*()]+)",
                    # More specific pattern for Cruise AV with asterisks and bullet points
                    r"\*\s*\*\*Auto(?:mated|nomous) Vehicle:\*\*\s*(\d{4})?\s*([A-Za-z][A-Za-z0-9\s\-]+)\s+AV(?:\.|\,|\s+The|\s+was|\s+operating)",
                ]
                for pattern in extraction_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)  # Added IGNORECASE for better matching
                    if match:
                        groups = match.groups()
                        if len(groups) >= 2:
                            if pattern.startswith(r"Auto(?:mated|nomous) Vehicle:(?:\s*\*\*)?\s*([A-Za-z]+)\s+([A-Za-z0-9\s]+?)"):
                                # Handle fallback pattern: Brand Model
                                brand_av = groups[0].strip() if groups[0] else "Unknown"
                                model_av = groups[1].strip() if groups[1] else "Unknown"
                            else:
                                # Handle original patterns
                                if groups[0]:
                                    year_av = groups[0]
                                if groups[1]:
                                    brand_av = groups[1].strip()
                                if len(groups) >= 3 and groups[2]:
                                    model_av = groups[2].strip()
                        break
            
            # Improved post-processing for models
            if model_av:
                # Remove common suffixes that might be part of the description
                model_av = re.sub(r'(?i)\s+(?:operating in|in|the report indicates|was|with|driving|on|that|indicated|mode|autonomous mode|automated mode|conventional mode)\s+.*$', '', model_av)
                
                # Handle special cases for well-known models
                if brand_av == "Cruise":
                    model_av = "AV"
                elif brand_av == "Tesla" and re.search(r'(?i)model\s*[3xyse]', model_av):
                    # Extract the model letter for Tesla models
                    model_match = re.search(r'(?i)model\s*([3xyse])', model_av)
                    if model_match:
                        model_letter = model_match.group(1).upper()
                        model_av = f"Model {model_letter}"
                
                # Handle special cases like "AV" appended to model names
                model_av = re.sub(r'(?i)\s+AV$', '', model_av)
                # Clean up extra whitespace
                model_av = re.sub(r'\s+', ' ', model_av).strip()
                # If model contains brand name at the beginning, remove it
                if brand_av and brand_av.lower() != "unknown" and model_av.lower().startswith(brand_av.lower()):
                    model_av = model_av[len(brand_av):].strip()
            
            # Get brand mappings from config
            brand_mapping = mappings.get("brand_mapping", {})
            av_company_mapping = mappings.get("av_company_mapping", {})
            model_mapping = mappings.get("model_mapping", {})
            
            # Normalize brand
            if brand_av:
                brand_lower = brand_av.lower()
                for key, value in brand_mapping.items():
                    if key in brand_lower:
                        brand_av = value
                        break
            # Handle specific model normalization
            if model_av:
                model_lower = model_av.lower()
                for key, value in model_mapping.items():
                    if key == model_lower:
                        model_av = value
                        break
                # If brand is Tesla and model contains "model", normalize it
                if brand_av == "Tesla" and "model" in model_lower:
                    if "3" in model_lower: model_av = "Model 3"
                    elif "x" in model_lower: model_av = "Model X"
                    elif "s" in model_lower: model_av = "Model S"
                    elif "y" in model_lower: model_av = "Model Y"
            # Set defaults if not detected
            if not brand_av: brand_av = "Unknown"
            if not model_av: model_av = "Unknown"
            if not year_av: year_av = "Unknown"
            
            # Return fetched values
            return [brand_av, model_av, year_av]
        
        elif q == "q2-av_mode":
            # Extract whether the AV was operating in automated mode or conventional mode
            pattern_autonomous = re.compile(r'(autonomous mode|automated mode|self-driving mode|operating in (?:autonomous|automated|self-driving) mode|was in (?:autonomous|automated|self-driving) mode)', re.IGNORECASE)
            pattern_conventional = re.compile(r'(conventional mode|manual mode|operating in conventional mode|was in conventional mode|human-driven|manually operated)', re.IGNORECASE)
            
            if pattern_autonomous.search(response):
                return "Yes"
            elif pattern_conventional.search(response):
                return "No"
            else:
                return "Unknown"
            
        elif q == "q2-other_road_user":
            # Standardize format by replacing different variants with a consistent format
            response = re.sub(
                r"Other Involved Road User:|The other involved party was a|The other involved road user was the|" +
                r"other party was a|road user was a|Other Involved Party:", 
                "Other Road User:", 
                response
            )
            response = re.sub(r"\*\*Other Road User:\*\*", "Other Road User:", response)
            
            # Check for specific manual cases first
            if "a pedestrian was also involved" in response or "Pedestrian:" in response:
                return "Pedestrian"
            elif "2013 Honda Civic" in response:
                return "Driver in vehicle"
                
            # Define road user categories and their associated keywords
            road_user_types = {
                "Pedestrian": ["pedestrian", "walking", "male", "female"],
                "Cyclist": ["bicyclist", "bicycle", "cyclist"],
                "Scooter": ["scooter", "moped"],
                "Motorcycle": ["motorcycle", "motorbike"],
                "Fixed object": ["fixed object", "parked car", "parked vehicle", "stationary"],
                "Driver in vehicle": ["driver"]
            }
            
            # Extract the road user description
            ru_match = re.search(r"Other Road User: \s*([^\n]*)", response)
            
            # Check if the description matches any of the defined categories
            if ru_match:
                description = ru_match.group(1).lower()
                for user_type, keywords in road_user_types.items():
                    if any(keyword in description for keyword in keywords):
                        return user_type
                return ru_match.group(1).strip()  # Return exact match if no category matched
            
            # Check for alternative formats mentioning vehicles
            response = re.sub(r"Vehicle 2:|Other Car:", "Other Vehicle:", response)
            if "Other Vehicle:" in response:
                return "Driver in vehicle"
                
            # No match found
            logger.debug(f"{row_index} q2-other_road_user: no match found for {response}.")
            return "Unknown"
        
        elif q == "q2-other_vehicle":
            # Load brand data from JSON
            json_path = os.path.join(os.path.dirname(__file__), 'av_brands.json')
            with open(json_path, 'r') as f:
                brand_data = json.load(f)
            
            # Get mappings from JSON
            brand_mapping = brand_data.get('brand_mapping', {})
            model_mapping = brand_data.get('model_mapping', {})
            specific_patterns = brand_data.get('specific_vehicle_patterns', {})
            
            # Define AV companies to exclude from "other vehicle" classification
            av_companies = ["Waymo", "Cruise", "Zoox", "Argo AI", "Aurora", "Mobileye", "Baidu", "Pony.ai", "Motional", "Apple", "Mosaic"]
            # Clean up response text
            response = re.sub(r"Vehicle 2:|Hyundai:|Other Car:", "Other Vehicle:", response)
            response = re.sub(r"\*\*Other Vehicle:\*\*", "Other Vehicle:", response)
            response = re.sub(r"(Make|Brand|Model|Year):", r"\1", response)
            response = re.sub(r"A 2023:", "2023", response)
            
            # Check for no vehicle phrases
            if any(phrase in response for phrase in [
                "No other vehicles were involved", "No other vehicle was involved", 
                "A truck is listed", "Not applicable", "A pedestrian", "A bicycle", 
                "A bicyclist", "None", "None listed", "N/A - only one vehicle"
            ]):return None
                
            # Check specific vehicle patterns
            for pattern, info in specific_patterns.items():
                if re.search(pattern, response, re.IGNORECASE):
                    brand = info["brand"]
                    # Skip if this is an AV company (not appropriate as "other vehicle")
                    if brand in av_companies:
                        continue
                    parts = [p for p in [info["year"], brand, info["model"]] if p]
                    return " ".join(parts)
            
            # Try standard pattern: Year, brand, model
            ov_match = re.search(r"Other Vehicle: Year (UNK|Unknown|\d{4}), Brand ([A-Za-z-]+), Model ([A-Za-z0-9\s]+)", response)
            if ov_match:
                brand = brand_mapping.get(ov_match.group(2).strip().lower(), ov_match.group(2).capitalize())
                # Skip if this is an AV company
                if brand in av_companies:
                    return "Unknown"
                model = model_mapping.get(ov_match.group(3).strip().lower(), ov_match.group(3))
                return f"{brand} {model}".strip()
                
            # Try simpler pattern
            ov_match_2 = re.search(r"Other Vehicle:\s*(\d{4})?\s*([^.,*()]+)", response)
            if ov_match_2:
                vehicle = ov_match_2.group(2).strip()
                # Check if the vehicle contains any AV company name
                if any(av_company.lower() in vehicle.lower() for av_company in av_companies):
                    return "Unknown"
                
                # Normalize brand names using mapping
                vehicle_lower = vehicle.lower()
                for brand_key, brand_value in brand_mapping.items():
                    # Skip AV companies in brand mapping
                    if brand_value in av_companies:
                        continue
                    if brand_key in vehicle_lower:
                        vehicle = re.compile(re.escape(brand_key), re.IGNORECASE).sub(brand_value, vehicle)
                        break
                # Clean up extracted text
                if vehicle in ["Unknown", "A pickup truck", "Year and model are unknown", ""]:
                    return "Unknown"
                # Manual filtering for specific vehicle types
                vehicle = re.sub(r"The (autonomous|automated) vehicle.*|A (Toyota|Nissan).*|pickup truck.*|truck.*", r"\2", vehicle, flags=re.DOTALL).strip()
                return vehicle
                
            # No match found
            logger.debug(f"q2-other_vehicle: no match found for {response}.")
            return None
        
        # New NLTK-based implementation starts here for Q3-Q7
        # Skip processing if empty response
        if not response or len(response.strip()) == 0:
            return "Unknown" if q not in ["q6-av_at_fault", "q6-contributing_factors", 
                                          "q7-traffic_conditions", "q7-av_movement", 
                                          "q7-other_road_user_movement", "q7-same_direction", 
                                          "q7-same_lane"] else None
        
        # Preprocessing to normalize and clean text
        def preprocess_text(text):
            # Replace markdown formatting
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        
        # Extract section for a specific question (Q1-Q7)
        def extract_question_section(text, q_num):
            pattern = rf"Q{q_num}\.?\s+.*?(?=Q{q_num+1}|$)"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(0)
            return text  # Fallback to full text if section not found
        
        # Extract value for a labeled field 
        def extract_labeled_value(text, field_name):
            # Try both markdown and plain text formats
            patterns = [
                rf"{field_name}:\s*([^,;\n*]+)",
                rf"\*\*{field_name}:\*\*\s*([^,;\n*]+)"
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match and match.group(1):  # Ensure match is not None and group exists
                    value = match.group(1).strip()
                    if value.lower() in ['unknown', 'not specified', 'n/a', 'none', '']:
                        return "Unknown"
                    return value
            return "Unknown"
        
        # Function to check if text indicates "not specified"
        def is_not_specified(text):
            not_specified_terms = ['not specified', 'unknown', 'n/a', 'none', 
                                   'unspecified', 'blank', 'empty', 'not provided']
            return any(term in text.lower() for term in not_specified_terms)
            
        # Clean response
        cleaned_response = preprocess_text(response)
        
        # NLTK-based implementations for Q3-Q7
        if q == "q3-address":
            q3_section = extract_question_section(cleaned_response, 3)
            address = extract_labeled_value(q3_section, "Address")
            return address
        
        elif q == "q3-street_type":
            q3_section = extract_question_section(cleaned_response, 3)
            street_type = extract_labeled_value(q3_section, "Street Type")
            return street_type
        
        elif q == "q3-lanes":
            q3_section = extract_question_section(cleaned_response, 3)
            lanes = extract_labeled_value(q3_section, "Lanes( & Width)?")
            return lanes
        
        elif q == "q3-area_type":
            q3_section = extract_question_section(cleaned_response, 3)
            area_type = extract_labeled_value(q3_section, "Urban/Rural")
            
            # Search for keywords anywhere in the text
            if "urban" in q3_section.lower():
                return "Urban"
            elif "rural" in q3_section.lower():
                return "Rural"
            
            # Return "Unknown" if no keywords found
            return "Unknown"
        
        elif q == "q3-coordinates":
            q3_section = extract_question_section(cleaned_response, 3)
            coords = extract_labeled_value(q3_section, "Google Maps Coordinates")
            if is_not_specified(coords):
                return "Unknown"
            return coords
        
        elif q == "q4-weather":
            q4_section = extract_question_section(cleaned_response, 4)
            
            # Define comprehensive weather categories with expanded keywords
            weather_categories = {
                "CLEAR": ["clear", "sunny", "fair", "good", "fine", "nice"],
                "CLOUDY": ["cloud", "cloudy", "overcast", "partly", "gray", "grey"],
                "RAINING": ["rain", "raining", "rainy", "shower", "drizzle", "wet", "precipitation"],
                "SNOWING": ["snow", "snowing", "snowy", "sleet", "hail", "icy", "ice", "freezing"],
                "FOG/VISIBILITY": ["fog", "foggy", "mist", "misty", "haze", "hazy", "visibility", "poor visibility"],
                "WIND": ["wind", "windy", "gust", "gusty", "storm", "stormy", "hurricane", "tornado"]
            }
            
            # Try direct labeled extraction first
            weather = extract_labeled_value(q4_section, "Weather")
            
            # If direct extraction worked, categorize it
            if weather != "Unknown":
                if is_not_specified(weather):
                    return "Unknown"
                    
                # Check which category it falls into
                for category, keywords in weather_categories.items():
                    if any(keyword in weather.lower() for keyword in keywords):
                        return category
            
            # Fallback to scanning entire section for weather keywords
            tokens = word_tokenize(q4_section.lower())
            for category, keywords in weather_categories.items():
                if any(keyword in tokens for keyword in keywords):
                    return category
            
            return "Unknown"
        
        elif q == "q4-lighting":
            q4_section = extract_question_section(cleaned_response, 4)
            
            # Define lighting categories with expanded keywords
            lighting_categories = {
                "DAYLIGHT": ["daylight", "day", "bright", "sunny", "clear day"],
                "DARK-STREET LIGHTS": ["street light", "lit", "dark with light", "dark with street light"],
                "DARK-NO STREE LIGHTS": ["no street light", "unlit", "dark with no light"],
                "DARK-STREE LIGHTS NOT FUNCTIONING": ["not functioning", "non-functioning", "broken light"],
                "DUSK": ["dusk", "dawn", "twilight", "sunset", "sunrise"]
            }
            
            # Try direct labeled extraction first
            lighting = extract_labeled_value(q4_section, "Lighting Conditions?")
            
            # If direct extraction worked, categorize it
            if lighting != "Unknown":
                if is_not_specified(lighting):
                    return "Unknown"
                
                # Default classification for just "dark"
                if lighting.lower() == "dark":
                    return "DARK-STREET LIGHTS"
                
                # Check which category it falls into
                for category, keywords in lighting_categories.items():
                    if any(keyword in lighting.lower() for keyword in keywords):
                        return category
            
            # Fallback to scanning entire section for lighting keywords
            for category, keywords in lighting_categories.items():
                if any(keyword in q4_section.lower() for keyword in keywords):
                    return category
            
            return "Unknown"
        
        elif q == "q4-surface":
            q4_section = extract_question_section(cleaned_response, 4)
            
            # Define surface categories with keywords
            surface_categories = {
                "DRY": ["dry", "normal", "clean"],
                "WET": ["wet", "damp", "moist", "water"],
                "SNOWY -ICY": ["snow", "snowy", "ice", "icy", "sleet", "frost", "frozen"],
                "SLIPPERY": ["slippery", "slick", "greasy", "oil"]
            }
            
            # Find the line with road surface information
            lines = q4_section.split('\n')
            for line in lines:
                if "road surface" in line.lower():
                    # Found the road surface line, now check for keywords
                    line_lower = line.lower()
                    for category, keywords in surface_categories.items():
                        if any(keyword in line_lower for keyword in keywords):
                            return category
                    break  # Stop after finding the first matching line
            # If no match was found or no surface line, return Unknown
            return "Unknown"
            
        elif q == "q4-conditions":
            q4_section = extract_question_section(cleaned_response, 4)
            conditions = extract_labeled_value(q4_section, "Road Conditions?")
            
            if conditions != "Unknown" and not is_not_specified(conditions):
                return conditions
            return "Unknown"
        
        elif q == "q5-collision_type":
            q5_section = extract_question_section(cleaned_response, 5)
            
            # Define primary collision categories with keywords
            collision_types = {
                "Pedestrian": ["pedestrian", "person", "foot", "walking"],
                "Cyclist": ["cyclist", "bicycle", "bike", "bicyclist"],
                "Rear-end": ["rear-end", "rear end", "from behind", "behind"],
                "Head-on": ["head-on", "head on", "frontal", "front"],
                "Sideswipe": ["sideswipe", "side swipe", "side collision", "side impact"],
                "Hit object": ["fixed object", "stationary object", "pole", "tree", "barrier", "parked"],
                "Broadside": ["broadside", "t-bone", "broadside"],
                "Other": ["other", "unknown", "not specified", "n/a", "none"]
            }
            
            # Find the line with collision type information
            lines = q5_section.split('\n')
            for line in lines:
                if "type of collision" in line.lower():
                    # Found the collision type line, check for keywords
                    line_lower = line.lower()
                    
                    # Skip if it's just a single letter (likely a code)
                    if re.search(r':\s*[a-z](\s|$|\.)', line_lower):
                        break
                    
                    # Extract only the part immediately after "collision:" or "collision"
                    collision_value = ""
                    match = re.search(r'collision:?\s+([^.,;*]+)', line_lower)
                    if match:
                        collision_value = match.group(1).strip()
                    
                    # Check for main categories using the extracted value
                    for category, keywords in collision_types.items():
                        # First check exact matches in the extracted value
                        if collision_value and any(keyword == collision_value for keyword in keywords):
                            return category
                        # Then check keyword containment
                        if collision_value and any(keyword in collision_value for keyword in keywords):
                            return category
                        # As fallback, check if keywords appear in the line
                        if any(f" {keyword} " in f" {line_lower} " for keyword in keywords):
                            return category
                    
                    # If extracted value exists and no category match, return the value itself
                    if collision_value and len(collision_value) > 1 and not any(term in collision_value for term in ["unknown", "none", "not specified", "n/a"]):
                        # Get only the first few words (max 3) to avoid long phrases
                        words = collision_value.split()
                        if len(words) > 3:
                            words = words[:3]
                        return " ".join(words).title()
                    
                    break
            
            # If no match found
            return "Other"
        
        elif q == "q5-av_damage":
            q5_section = extract_question_section(cleaned_response, 5)
            
            # Define standard damage categories with keywords
            damage_categories = {
                "Minor": ["minor", "slight", "minimal", "cosmetic", "small"],
                "Moderate": ["moderate", "medium", "dent", "damaged", "visible"],
                "Severe": ["severe", "major", "heavy", "extensive", "significant"],
                "None": ["none", "no damage", "not damaged", "undamaged"],
            }
            
            # Find lines mentioning vehicle damage
            lines = q5_section.split('\n')
            for line in lines:
                if "vehicle damage" in line.lower() or "damage" in line.lower() and any(term in line.lower() for term in ["av", "autonomous", "automated", "tesla"]):
                    line_lower = line.lower()
                    
                    # Check for standard damage categories
                    for category, keywords in damage_categories.items():
                        if any(keyword in line_lower for keyword in keywords):
                            return category
                    
                    # If we didn't find a standard category but found damage info
                    # Just return a generic "Damaged" rather than the full text
                    if "damage" in line_lower:
                        return "Damaged"
                    
                    break
            
            # If no match found, return Other
            return "Other"
        
        elif q == "q5-av_damage_category":
            # This function is now merged with q5-av_damage
            # For backward compatibility, call the q5-av_damage function
            return self.categorise(response, "q5-av_damage", row_index)
        
        elif q == "q5-injuries":
            q5_section = extract_question_section(cleaned_response, 5)
            
            # Look for injury information in the section
            injuries_info = extract_labeled_value(q5_section, "Injuries(/Deaths/Property Damage)?")
            
            # Check if injuries are mentioned
            if injuries_info != "Unknown":
                if "no injur" in injuries_info.lower():
                    return "No"
                elif "injur" in injuries_info.lower():
                    return "Yes"
            
            # Fallback to scanning the entire section
            if "no injur" in q5_section.lower():
                return "No"
            elif "injur" in q5_section.lower() and "no injur" not in q5_section.lower():
                return "Yes"
            
            return "Unknown"
        
        elif q == "q6-av_at_fault":
            q6_section = extract_question_section(cleaned_response, 6)
            sentences = sent_tokenize(q6_section.lower())
            
            for sentence in sentences:
                # Check for AV at fault
                if any(term in sentence for term in ["automated vehicle is at fault", 
                                                    "autonomous vehicle is at fault",
                                                    "av is at fault", 
                                                    "automated vehicle was at fault",
                                                    "autonomous vehicle was at fault",
                                                    "av was at fault"]):
                    return True
                # Check for other party at fault
                elif any(term in sentence for term in ["other road user is at fault",
                                                      "pedestrian is at fault",
                                                      "driver is at fault",
                                                      "other road user was at fault",
                                                      "pedestrian was at fault",
                                                      "driver was at fault"]):
                    return False
            
            # Fallback to keyword proximity analysis
            tokens = word_tokenize(q6_section.lower())
            av_index = -1
            fault_index = -1
            
            for i, token in enumerate(tokens):
                if token in ["av", "autonomous", "automated"]:
                    av_index = i
                elif token in ["fault", "responsible", "guilty"]:
                    fault_index = i
            
            if av_index >= 0 and fault_index >= 0:
                # Check if they're close to each other (within 5 words)
                if abs(av_index - fault_index) <= 5:
                    return True
            
            return None
        
        elif q == "q6-contributing_factors":
            q6_section = extract_question_section(cleaned_response, 6)
            
            # Try to extract contributing factors
            factors_info = extract_labeled_value(q6_section, "Contributing Factors?")
            
            if factors_info != "Unknown":
                # Split by common separators
                factors = re.split(r',|\band\b|;', factors_info)
                return [factor.strip() for factor in factors if factor.strip()]
            
            # Look for quoted factors as fallback
            factors_match = re.search(r'Contributing Factors:.*?\"([^\"]+)\"', q6_section)
            if factors_match:
                factors = factors_match.group(1).split(',')
                return [factor.strip() for factor in factors]
                
            return None
        
        elif q == "q7-traffic_conditions":
            q7_section = extract_question_section(cleaned_response, 7)
            
            # Define standard traffic condition categories
            traffic_conditions = {
                "Heavy": ["heavy", "congested", "busy", "dense", "high volume", "backed up", "bumper to bumper"],
                "Moderate": ["moderate", "medium", "regular", "normal", "average"],
                "Light": ["light", "minimal", "low", "sparse", "little", "not busy", "calm"],
                "None": ["no traffic", "empty", "clear", "none", "nonexistent", "vacant"]
            }
            
            # Find lines mentioning traffic
            lines = q7_section.split('\n')
            for line in lines:
                if "traffic" in line.lower():
                    line_lower = line.lower()
                    
                    # Check for standard traffic conditions
                    for condition, keywords in traffic_conditions.items():
                        if any(keyword in line_lower for keyword in keywords):
                            return condition
                    break
            
            # If no match found
            return "Unknown"
        
        elif q == "q7-av_movement":
            q7_section = extract_question_section(cleaned_response, 7)
            
            # Define standard movement categories
            movement_categories = {
                "Moving forward": ["moving forward", "traveling", "driving", "proceeding", "moving straight", "going forward"],
                "Stopped": ["stopped", "stationary", "parked", "idle", "not moving", "standing"],
                "Turning": ["turning", "making a turn", "turning left", "turning right", "changing direction"],
                "Slowing": ["slowing", "braking", "decelerating", "coming to a stop", "reducing speed"],
                "Accelerating": ["accelerating", "speeding up", "increasing speed"],
                "Reversing": ["reversing", "backing up", "backing", "in reverse"]
            }
            
            # Find lines mentioning AV movement
            lines = q7_section.split('\n')
            for line in lines:
                line_lower = line.lower()
                if any(term in line_lower for term in ["autonomous vehicle", "automated vehicle", "av", "tesla"]) and any(movement in line_lower for movement in ["moving", "driving", "traveling", "stopped", "turning", "proceeding"]):
                    # Check for standard movement categories
                    for category, keywords in movement_categories.items():
                        if any(keyword in line_lower for keyword in keywords):
                            return category
                    
                    # If we found a line with movement but no standard category,
                    # just return "Moving" as a generic fallback
                    return "Moving"
            
            # If no match found
            return "Unknown"
        
        elif q == "q7-other_road_user_movement":
            q7_section = extract_question_section(cleaned_response, 7)
            
            # Define standard movement categories for other road users
            movement_categories = {
                "Moving": ["moving", "walking", "traveling", "driving", "proceeding", "running", "jogging"],
                "Stopped": ["stopped", "stationary", "standing", "idle", "not moving", "waiting"],
                "Crossing": ["crossing", "crossing street", "crossing road", "in crosswalk", "crossing intersection", "jaywalking"],
                "Turning": ["turning", "making a turn", "turning left", "turning right"],
                "In traffic": ["in traffic", "among traffic", "in roadway", "in the road", "in the street"]
            }
            
            # Find lines mentioning other road user
            lines = q7_section.split('\n')
            for line in lines:
                line_lower = line.lower()
                if any(term in line_lower for term in ["pedestrian", "cyclist", "bicyclist", "other road user", "other party"]):
                    # Check for standard movement categories
                    for category, keywords in movement_categories.items():
                        if any(keyword in line_lower for keyword in keywords):
                            return category
                    
                    # If we found a line mentioning the other road user but no standard movement,
                    # try to extract just the verb following the user type
                    match = re.search(r'(pedestrian|cyclist|bicyclist|other road user|other party)\s+(?:was|is)?\s+(\w+ing)', line_lower)
                    if match:
                        return match.group(2).capitalize()
            
            # If no match found
            return "Unknown"
        
        elif q == "q7-same_direction":
            q7_section = extract_question_section(cleaned_response, 7)
            
            # Check for direction information
            if "same direction" in q7_section.lower():
                return True
            elif "different direction" in q7_section.lower() or "opposite direction" in q7_section.lower():
                return False
            
            return None
        
        elif q == "q7-same_lane":
            q7_section = extract_question_section(cleaned_response, 7)
            
            # Check for lane information
            if "same lane" in q7_section.lower():
                return True
            elif "different lane" in q7_section.lower():
                return False
            
            return None
        
        else:
            # For any other question types, return the default
            logger.debug(f"{row_index} {q}: unrecognized question type")
            return "wrong question"

    def process_answers(self, df):
        """Apply categorisation to each of the questions in the query."""
        logger.info('Processing output.')
        
        # for i in range(1, num_q + 1):
        #     question_col = f"q{i}"
        #     if question_col not in df.columns:
        #         df[question_col] = df["response"].apply(lambda x: self.extract_answers(str(x))[question_col])
        #     df[f"q{i}_category"] = df[question_col].apply(lambda x: self.categorise_response(str(x)))
        # Q1
        df["q1"] = df.apply(lambda row: self.extract_answers(str(row["response"]), 1)["q1"], axis=1)
        df["q1_category"] = df.apply(lambda row: self.categorise(str(row["q1"]), "q1", row.name), axis=1)

        # Q2
        df["q2"] = df.apply(lambda row: self.extract_answers(str(row["response"]), 2)["q2"], axis=1)
        df[["q2_av_brand", "q2_av_model", "q2_av_year"]] = df.apply(
            lambda row: pd.Series(self.categorise(str(row["q2"]), "q2-av", row.name)), axis=1
        )
        df["q2_av_mode"] = df.apply(lambda row: self.categorise(str(row["q2"]), "q2-av_mode", row.name), axis=1)  # noqa: E501
        df["q2_other_road_user"] = df.apply(lambda row: self.categorise(str(row["q2"]), "q2-other_road_user", row.name), axis=1)  # noqa: E501
        df["q2_other_vehicle"] = df.apply(lambda row: self.categorise(str(row["q2"]), "q2-other_vehicle", row.name), axis=1)  # noqa: E501

        # Q3
        df["q3"] = df.apply(lambda row: self.extract_answers(str(row["response"]), 3)["q3"], axis=1)
        df["q3_address"] = df.apply(lambda row: self.categorise(str(row["q3"]), "q3-address", row.name), axis=1)
        df["q3_street_type"] = df.apply(lambda row: self.categorise(str(row["q3"]), "q3-street_type", row.name), axis=1)
        df["q3_lanes"] = df.apply(lambda row: self.categorise(str(row["q3"]), "q3-lanes", row.name), axis=1)
        df["q3_area_type"] = df.apply(lambda row: self.categorise(str(row["q3"]), "q3-area_type", row.name), axis=1)
        df["q3_coordinates"] = df.apply(lambda row: self.categorise(str(row["q3"]), "q3-coordinates", row.name), axis=1)

        # Q4 - First extract the raw answer
        df["q4"] = df.apply(lambda row: self.extract_answers(str(row["response"]), 4)["q4"], axis=1)
        # Then process each environmental condition
        df["q4_weather"] = df.apply(lambda row: self.categorise(str(row["q4"]), "q4-weather", row.name), axis=1)
        df["q4_lighting"] = df.apply(lambda row: self.categorise(str(row["q4"]), "q4-lighting", row.name), axis=1)
        df["q4_surface"] = df.apply(lambda row: self.categorise(str(row["q4"]), "q4-surface", row.name), axis=1)
        df["q4_conditions"] = df.apply(lambda row: self.categorise(str(row["q4"]), "q4-conditions", row.name), axis=1)
        
        # Q5
        df["q5"] = df.apply(lambda row: self.extract_answers(str(row["response"]), 5)["q5"], axis=1)
        df["q5_collision_type"] = df.apply(lambda row: self.categorise(str(row["q5"]), "q5-collision_type", row.name), axis=1)
        # Use the combined AV damage field and remove the redundant category
        df["q5_av_damage"] = df.apply(lambda row: self.categorise(str(row["q5"]), "q5-av_damage", row.name), axis=1)
        df["q5_other_vehicle_damage"] = df.apply(lambda row: self.categorise(str(row["q5"]), "q5-other_vehicle_damage", row.name), axis=1)
        df["q5_injuries"] = df.apply(lambda row: self.categorise(str(row["q5"]), "q5-injuries", row.name), axis=1)

        # Q6
        df["q6"] = df.apply(lambda row: self.extract_answers(str(row["response"]), 6)["q6"], axis=1)
        df["q6_av_at_fault"] = df.apply(lambda row: self.categorise(str(row["q6"]), "q6-av_at_fault", row.name), axis=1)
        df["q6_contributing_factors"] = df.apply(lambda row: self.categorise(str(row["q6"]), "q6-contributing_factors", row.name), axis=1)

        # Q7
        df["q7"] = df.apply(lambda row: self.extract_answers(str(row["response"]), 7)["q7"], axis=1)
        df["q7_traffic_conditions"] = df.apply(lambda row: self.categorise(str(row["q7"]), "q7-traffic_conditions", row.name), axis=1)
        df["q7_av_movement"] = df.apply(lambda row: self.categorise(str(row["q7"]), "q7-av_movement", row.name), axis=1)
        df["q7_other_road_user_movement"] = df.apply(lambda row: self.categorise(str(row["q7"]), "q7-other_road_user_movement", row.name), axis=1)
        df["q7_same_direction"] = df.apply(lambda row: self.categorise(str(row["q7"]), "q7-same_direction", row.name), axis=1)
        df["q7_same_lane"] = df.apply(lambda row: self.categorise(str(row["q7"]), "q7-same_lane", row.name), axis=1)

        return df

    def pdf_to_base64_image(self, file, resize_image=False, resize_dimentions=(896, 896)):
        """Turn pages of the PDF file with the report to base64 strings.
        Args:
            file (str): Name of file of the report.

        Returns:
            base64_image (list): List of pages as base64 strings.
        """
        # create full path of the file with the report
        file = os.fsdecode(file)
        full_path = os.path.join(self.files_reports, file)
        # each page is 1 base64_image
        base64_images = []
        imgs = convert_from_path(full_path)
        temp_png = 'output_images'
        if not os.path.exists(temp_png):
            os.makedirs(temp_png)
        for i, image in enumerate(imgs):
            # save generated images. This can be overwritten.
            image_path = os.path.join(temp_png, f"page_{i+1}.png")
            # resize image with preserving the aspect ratio
            if (resize_image):
                image.thumbnail(resize_dimentions, Image.Resampling.LANCZOS)
            # save image
            image.save(image_path, 'PNG')
            base64_images.append(self.encode_image(image_path))
        # close image
        logger.debug('Turned report {} into base64 images.', file)
        # combine all base64 images into one string
        # base64_images = ''.join(base64_images)
        return base64_images

    def encode_image(self, image_path):
        """Return base64 string for an image.
        Args:
            image_path (TYPE): Path of image.

        Returns:
            str: encoded string.
        """
        with open(image_path, "rb") as imageFile:
            return base64.b64encode(imageFile.read()).decode('utf-8')

    def ask_llm(self, file, pages):
        """Receive responses from LLM API for all pages at once.
        Args:
            file (str): File with report.
            pages (list): List of pages as base64 strings.

        Returns:
            dataframe: dataframe with responses.
        """
        # build content with multiple images
        # first add a query to the content list
        content = [{
                    "type": "text",
                    "text": llme.common.get_configs('query'),
                    }
                   ]
        # populate the list with base64 strings of pages in the report
        for page in pages:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{page}",
                    "detail": "high"
                    },
                })
        # object to store response
        response = None
        # send request to GPT4-V
        try:
            response = self.gpt_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                    ],
            )
            logger.debug('Received response from LLM: {}.', response.choices[0])
        except openai.AuthenticationError:
            logger.error('Incorrect API key.')
            return None
        except openai.BadRequestError as e:
            logger.error('Bad request given: {}.', e)
            return None
        except openai.RateLimitError:
            logger.warning('Rate limit exceeded. Retrying after a short delay...')
            time.sleep(60)  # wait 60 seconds
            return self.ask_llm(file, pages)
        except Exception as e:
            logger.error(
                f"\nError chatting with the LM Studio server!\n\n"
                f"Please ensure:\n"
                f"1. LM Studio server is running at 127.0.0.1:1234 (hostname:port)\n"
                f"2. Model '{MODEL}' is downloaded\n"
                f"3. Model '{MODEL}' is loaded, or that just-in-time model loading is enabled\n\n"
                f"Error details: {str(e)}\n"
                "See https://lmstudio.ai/docs/basics/server for more information"
            )
            exit(1)
        # turn response into a dataframe
        data = {'report': [file], 'response': [response.choices[0].message.content]}
        df = pd.DataFrame(data)
        return df

    def filter_data(self, df):
        """
        Filter data.
        Args:
            df (dataframe): dataframe with data.

        Returns:
            dataframe: updated dataframe.
        """
        logger.error('Filtering data not implemented.')
        # assign to attribute
        self.chatgpt_data = df
        # return df with data
        return df

    def clean_data(self, df):
        """Clean data from unexpected values.

        Args:
            df (dataframe): dataframe with data.

        Returns:
            dataframe: updated dataframe.
        """
        logger.error('Cleaning data not implemented.')
        # assign to attribute
        self.chatgpt_data = df
        # return df with data
        return df

    def show_info(self):
        """
        Output info for data in object.
        """
        logger.info('No info to show.')
