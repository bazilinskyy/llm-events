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

import llmevents as llme

# warning about partial assignment
pd.options.mode.chained_assignment = None  # default='warn'

logger = llme.CustomLogger(__name__)  # use custom logger


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
        
        elif q == "q3-address":
            # Extract specific address information
            address_match = re.search(r'\*\*Address:\*\*\s*([^*\n.]+)', response)
            if address_match:
                return address_match.group(1).strip()
            return "Unknown"
            
        elif q == "q3-street_type":
            # Extract street type information
            street_match = re.search(r'\*\*Street Type:\*\*\s*([^*\n.]+)', response)
            if street_match:
                return street_match.group(1).strip()
            return "Unknown"
            
        elif q == "q3-lanes":
            # Extract lanes and width information
            lanes_match = re.search(r'\*\*Lanes & Width:\*\*\s*([^*\n.]+)', response)
            if lanes_match:
                return lanes_match.group(1).strip()
            return "Unknown"
            
        elif q == "q3-area_type":
            # Extract urban/rural information
            area_match = re.search(r'\*\*Urban/Rural:\*\*\s*([^*\n.]+)', response)
            if area_match:
                area_text = area_match.group(1).strip().lower()
                if "urban" in area_text:
                    return "Urban"
                elif "rural" in area_text:
                    return "Rural"
                else:
                    return area_match.group(1).strip()
            return "Unknown"
            
        elif q == "q3-coordinates":
            # Extract GPS coordinates if available
            coords_match = re.search(r'\*\*Google Maps Coordinates:\*\*\s*([^*\n.]+)', response)
            if coords_match:
                coords = coords_match.group(1).strip()
                if any(term in coords.lower() for term in ["not provided", "unknown", "n/a", "none"]):
                    return "Unknown"
                return coords
            return "Unknown"
        
        elif q == "q4-weather":
            # Define keyword mappings for different weather conditions
            weather_keywords = {
                "CLEAR": ["clear", "sunny", "fair", "good", "fine"],
                "CLOUDY": ["cloud", "overcast", "partly"],
                "RAINING": ["rain", "shower", "drizzle", "precipitation", "wet"],
                "SNOWING": ["snow", "sleet", "hail", "freezing", "icy", "ice"],
                "FOG/VISIBILITY": ["fog", "foggy", "mist", "misty", "haze", "hazy", "visibility", "poor visibility", "limited visibility"],
                "WIND": ["wind", "windy", "gust", "gusty", "storm", "stormy", "hurricane", "tornado"]
            }
            
            # Not specified terms
            not_specified = ['not specified', 'unknown', 'n/a', 'none', 'unspecified', 'blank', 'empty', 'not provided']
            
            # Clean formatting from entire response - remove special characters
            clean_response = re.sub(r'[\*•\-\[\]()]', '', response).lower()
            
            # Try to extract weather section between "Weather:" and the next section
            weather_section_match = re.search(r'weather:?\s*(.*?)(?=lighting|road|q\d|$)', clean_response, re.DOTALL | re.IGNORECASE)
            
            if weather_section_match:
                # Remove code prefixes like "A - " or "A: "
                weather_text = re.sub(r'^[a-z]\s*[-:]\s*', '', weather_section_match.group(1).strip())
                
                # Skip single letter responses
                if len(weather_text) <= 1:
                    pass
                # Check for "not specified" cases
                elif any(term in weather_text for term in not_specified):
                    return "Unknown"
                # Check for weather keywords
                else:
                    for category, keywords in weather_keywords.items():
                        for keyword in keywords:
                            # Use simple 'in' check rather than regex for better matching
                            if keyword in weather_text:
                                return category
                    # If we have text but no keyword matched
                    return "OTHER"
            
            # Fallback: check the Q4 section for weather keywords
            q4_section = re.search(r'q4.+?(?=q5|$)', clean_response, re.DOTALL | re.IGNORECASE)
            search_text = q4_section.group(0) if q4_section else clean_response
            
            # Look for weather keywords in the search text
            for category, keywords in weather_keywords.items():
                for keyword in keywords:
                    # Use simple 'in' check instead of word boundaries for more matches
                    if keyword in search_text:
                        return category
            
            # If nothing found
            return "Unknown"
        
        elif q == "q4-lighting":
            # Define patterns and category mappings
            lighting_patterns = [
                # First try exact markdown format
                r'\*\s*\*\*Lighting\s*Conditions?\*\*:?\s*([^,;\n.]+)',
                # Then try the clean format
                r'Lighting\s*Conditions?:?\s*([^,;\n.]+)',
                # Then try other variations
                r'lighting(?:\s+(?:was|were|is|are))?\s*[:-]?\s*([^,.;\n]+)'
            ]
            
            lighting_categories = {
                # Dark - Street Lights Not Functioning patterns
                r'dark\s*-?\s*street\s*lights?\s*not\s*function|not\s*function|non[\s-]function': "DARK-STREE LIGHTS NOT FUNCTIONING",
                # Dark - No Street Lights patterns
                r'dark\s*-?\s*no\s*street\s*lights?|no street|no light|unlit': "DARK-NO STREE LIGHTS",
                # Dark - Street Lights patterns
                r'dark\s*-?\s*street\s*lights?|street light|streetlight|lit': "DARK-STREET LIGHTS",
                # Daylight patterns
                r'daylight|day light|daytime|day|sunny|sunlight|bright|clear day': "DAYLIGHT",
                # Dusk patterns
                r'dusk|twilight|dawn': "DUSK",
                # Dark (default to street lights)
                r'dark|night': "DARK-STREET LIGHTS"
            }
            
            # Special cases for not specified
            not_specified = ['not specified', 'unknown', 'n/a', 'none', 'unspecified', 'blank', 'empty', 'not provided']
            
            # Clean formatting and get Q4 section
            clean_response = re.sub(r'[\*•\-\[\]]', '', response)
            q4_section_match = re.search(r'Q4\.?\s+Time and environmental conditions.*?(?=Q5|$)', clean_response, re.DOTALL | re.IGNORECASE)
            q4_section = q4_section_match.group(0) if q4_section_match else clean_response
            
            # Handle "s, road surface" issue
            combined_pattern = re.search(r'Lighting\s*Conditions?:?\s*([^,;\n.]*)\s*,\s*Road\s*Surface', q4_section, re.IGNORECASE)
            if combined_pattern and combined_pattern.group(1).strip().lower() in ['s', '']:
                # Skip this and let the pattern matching handle it
                pass
            
            # Try to extract lighting text using the patterns
            lighting_text = None
            
            # First try in original response (especially for markdown)
            for pattern in lighting_patterns:
                lighting_match = re.search(pattern, response, re.IGNORECASE)
                if lighting_match:
                    candidate = lighting_match.group(1).strip().lower()
                    if candidate != 's':  # Skip the problematic 's' value
                        lighting_text = candidate
                        break
            
            # If not found, try in q4_section
            if not lighting_text:
                for pattern in lighting_patterns:
                    lighting_match = re.search(pattern, q4_section, re.IGNORECASE)
                    if lighting_match:
                        candidate = lighting_match.group(1).strip().lower()
                        if candidate != 's':  # Skip the problematic 's' value
                            lighting_text = candidate
                            break
            
            # Categorize the lighting text if found
            if lighting_text:
                # Check for "not specified" cases
                if any(term in lighting_text for term in not_specified):
                    return "Unknown" 
                # Check against category patterns
                for pattern, category in lighting_categories.items():
                    if re.search(pattern, lighting_text, re.IGNORECASE):
                        return category
                # If it's a single letter, return Unknown
                if re.match(r'^[a-zA-Z]$', lighting_text):
                    return "Unknown"
            # Fallback to keyword search in the whole section
            for pattern, category in lighting_categories.items():
                if re.search(pattern, q4_section, re.IGNORECASE):
                    return category
            # If all else fails, return Unknown
            return "Unknown"
        
        elif q == "q4-surface":
            # First clean formatting from response
            clean_response = re.sub(r'[\*•\-\[\]]', '', response)
            
            # Try to find the Q4 section for more targeted extraction
            q4_section_match = re.search(r'Q4\.?\s+Time and environmental conditions.*?(?=Q5|$)', clean_response, re.DOTALL | re.IGNORECASE)
            if q4_section_match:
                q4_section = q4_section_match.group(0)
            else:
                q4_section = clean_response
            
            # Extract road surface using more precise pattern within Q4 section
            surface_match = re.search(r'Road\s*Surface:?\s*([^,;\n.]+)', q4_section, re.IGNORECASE)
            
            if surface_match:
                surface_text = surface_match.group(1).strip().lower()
                # Check for "not specified" or missing values
                if any(term in surface_text for term in ['not specified', 'unknown', 'n/a', 'none', 'unspecified', 'blank', 'empty', 'not provided']):
                    return "Unknown" 
                # Map to standard surface categories
                if any(term in surface_text for term in ['dry', 'normal']):
                    return "DRY"
                elif any(term in surface_text for term in ['wet', 'damp', 'moist']):
                    return "WET"
                elif any(term in surface_text for term in ['snow', 'ice', 'icy', 'sleet', 'frost']):
                    return "SNOWY -ICY"
                elif any(term in surface_text for term in ['slippery', 'slick', 'greasy']):
                    return "SLIPPERY"
                else:
                    return "Unknown"
            return "Unknown"
        
        elif q == "q4-conditions":
            # First clean formatting from response
            clean_response = re.sub(r'[\*•\-\[\]]', '', response)
            
            # Try to find the Q4 section for more targeted extraction
            q4_section_match = re.search(r'Q4\.?\s+Time and environmental conditions.*?(?=Q5|$)', clean_response, re.DOTALL | re.IGNORECASE)
            if q4_section_match:
                q4_section = q4_section_match.group(0)
            else:
                q4_section = clean_response
            
            # Extract road conditions using more precise pattern within Q4 section
            conditions_match = re.search(r'Road\s*Conditions?:?\s*([^,;\n.]+)', q4_section, re.IGNORECASE)
            
            if conditions_match:
                conditions_text = conditions_match.group(1).strip()
                
                # Check for "not specified" or missing values
                if any(term in conditions_text.lower() for term in ['not specified', 'unknown', 'n/a', 'none', 'unspecified', 'blank', 'empty', 'not provided']):
                    return "Unknown"
                
                return conditions_text
            return "Unknown"

        elif q == "q5-collision_type":
            # Extract collision type directly from the formatted response
            collision_match = re.search(r'\*\*Type of Collision:\*\*\s*([^*\n.]+)', response)
            if not collision_match:
                # Try alternate format
                collision_match = re.search(r'Type of Collision:\s*([^*\n.]+)', response)
            if not collision_match:
                return "Unknown"
            # Get the raw collision text and clean it
            collision_text = collision_match.group(1).strip()
            # Remove quotation marks and other artifacts
            collision_text = re.sub(r'["""\'()]', '', collision_text)
            # If it's a single letter code, consider it unknown
            if re.match(r'^[A-Za-z0-9]$', collision_text):
                return "Unknown"
            
            # Priority order for collision types - check if these key terms appear
            priority_types = {
                "pedestrian": "Pedestrian",
                "cyclist": "Cyclist", 
                "bicycle": "Cyclist",
                "bicyclist": "Cyclist",
                "fixed object": "Fixed object",
                "rear-end": "Rear-end",
                "rear end": "Rear-end",
                "head-on": "Head-on", 
                "head on": "Head-on"
            }
            collision_lower = collision_text.lower()
            # Check for priority collision types first
            for key, value in priority_types.items():
                if key in collision_lower:
                    return value
            
            # For collision patterns with letter codes followed by descriptions (e.g., "A - Unsafe speed")
            pcf_match = re.match(r'^[A-Za-z0-9]\s*[-:]\s*(.+)', collision_text)
            if pcf_match:
                # Extract the description part
                collision_text = pcf_match.group(1).strip()
                # Check again for priority types in the description part
                collision_lower = collision_text.lower()
                for key, value in priority_types.items():
                    if key in collision_lower:
                        return value
            # If it's a very short description with known invalid values, return Unknown
            if collision_text.lower() in ["unknown", "none", "not applicable", "n/a"]:
                return "Unknown"
            # Check if this is a long description (e.g., describing vehicle movements)
            # Long descriptions typically have more than a few words and contain verbs like "was" or "traveling"
            if len(collision_text.split()) > 5 or re.search(r'\b(was|traveling|moving|making|turning|driving)\b', collision_text, re.IGNORECASE):
                return "Unknown"
                
            # Otherwise, simply return the text with proper capitalization for short, direct responses
            return collision_text.strip().title()
        
        elif q == "q5-av_damage":
            # Extract vehicle damage
            vehicle_damage_match = re.search(r'\*\*Vehicle Damage:\*\*\s*([^*]+)', response)
            if vehicle_damage_match:
                vehicle_damage = vehicle_damage_match.group(1).strip()
                
                # Look for autonomous vehicle damage
                av_damage_match = re.search(r'(Tesla|AV|automated vehicle|autonomous vehicle)[^.]*(damage[^.]*)', vehicle_damage, re.IGNORECASE)
                if av_damage_match:
                    return av_damage_match.group(2).strip()
            return "Unknown"
        
        elif q == "q5-av_damage_category":
            # Define damage categories for pattern matching
            damage_categories = {
                'minor': ['minor', 'slight', 'minimal', 'cosmetic'],
                'moderate': ['moderate', 'considerable', 'visible', 'damaged', 'dent'],
                'severe': ['severe', 'major', 'extensive', 'significant', 'heavy'],
                'total': ['total', 'destroyed', 'totaled']
            }
            
            # First get the AV damage description
            vehicle_damage_match = re.search(r'\*\*Vehicle Damage:\*\*\s*([^*]+)', response)
            if vehicle_damage_match:
                vehicle_damage = vehicle_damage_match.group(1).strip()
                
                # Look for autonomous vehicle damage
                av_damage_match = re.search(r'(Tesla|AV|automated vehicle|autonomous vehicle)[^.]*(damage[^.]*)', vehicle_damage, re.IGNORECASE)
                if av_damage_match:
                    av_damage = av_damage_match.group(2).strip()
                    
                    # Classify damage severity for AV
                    for category, keywords in damage_categories.items():
                        if any(keyword in av_damage.lower() for keyword in keywords):
                            return category
            return "unknown"
        
        elif q == "q5-other_vehicle_damage":
            # Extract vehicle damage
            vehicle_damage_match = re.search(r'\*\*Vehicle Damage:\*\*\s*([^*]+)', response)
            if vehicle_damage_match:
                vehicle_damage = vehicle_damage_match.group(1).strip()
                
                # Look for other vehicle damage
                other_damage_match = re.search(r'([^T]oyota|Honda|Ford|Chrysler|other vehicle)[^.]*(damage[^.]*)', vehicle_damage, re.IGNORECASE)
                if other_damage_match:
                    return other_damage_match.group(2).strip()
            return "Unknown"
        
        elif q == "q5-injuries":
            # Extract injuries information
            injuries_match = re.search(r'\*\*Injuries/Deaths/Property Damage:\*\*\s*([^*]+)', response)
            if injuries_match:
                injuries_text = injuries_match.group(1).strip()
                
                # Determine if there were injuries
                if re.search(r'injur(y|ies|ed)', injuries_text, re.IGNORECASE) and not re.search(r'no injur(y|ies|ed)', injuries_text, re.IGNORECASE):
                    return "Yes"
                elif 'no injuries' in injuries_text.lower():
                    return "No"
            return "Unknown"

        elif q == "q6-av_at_fault":
            # Extract if AV is at fault
            if re.search(r'(autonomous|automated|AV).*?\bat fault\b', response, re.IGNORECASE):
                return True
            elif re.search(r'(pedestrian|other road user|driver).*?\bat fault\b', response, re.IGNORECASE):
                return False
            else:
                return None
            
        elif q == "q6-contributing_factors":
            # Extract contributing factors
            contributing_factors = []
            
            # Look for quoted factors
            factors_match = re.search(r'Contributing Factors:.*?\"([^\"]+)\"', response)
            if factors_match:
                factors = factors_match.group(1).split(',')
                contributing_factors = [factor.strip() for factor in factors]
            else:
                # Alternative format without quotes
                factors_match = re.search(r'Contributing Factors:([^*]+)', response)
                if factors_match:
                    factors_text = factors_match.group(1).strip()
                    # Split by common separators
                    factors = re.split(r',|\band\b|;', factors_text)
                    contributing_factors = [factor.strip() for factor in factors if factor.strip()]
            
            return contributing_factors if contributing_factors else None

        elif q == "q7-traffic_conditions":
            # Extract traffic conditions
            traffic_match = re.search(r'\*\*Traffic:\*\*\s*([^*\n.]+)', response)
            if traffic_match:
                return traffic_match.group(1).strip()
            return None
            
        elif q == "q7-av_movement":
            # Extract AV movement
            av_movement_match = re.search(r'(autonomous|automated|AV).*?(traveling|moving|driving|stopped)([^.]*)', response, re.IGNORECASE)
            if av_movement_match:
                return (av_movement_match.group(2) + av_movement_match.group(3)).strip()
            return None
            
        elif q == "q7-other_road_user_movement":
            # Extract other road user movement
            other_movement_match = re.search(r'(pedestrian|other road user|driver).*?(walking|running|crossing|stopped|traveling|moving)([^.]*)', response, re.IGNORECASE)
            if other_movement_match:
                return (other_movement_match.group(2) + other_movement_match.group(3)).strip()
            return None
            
        elif q == "q7-same_direction":
            # Check if same direction
            same_direction_match = re.search(r'(same|different) direction', response, re.IGNORECASE)
            if same_direction_match:
                return 'same' in same_direction_match.group(1).lower()
            return None
            
        elif q == "q7-same_lane":
            # Check if same lane
            same_lane_match = re.search(r'(same|different) lanes?', response, re.IGNORECASE)
            if same_lane_match:
                return 'same' in same_lane_match.group(1).lower()
            return None
        else:
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
        df["q5_av_damage"] = df.apply(lambda row: self.categorise(str(row["q5"]), "q5-av_damage", row.name), axis=1)
        df["q5_av_damage_category"] = df.apply(lambda row: self.categorise(str(row["q5"]), "q5-av_damage_category", row.name), axis=1)
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
