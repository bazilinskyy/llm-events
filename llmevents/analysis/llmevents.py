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
        # pattern = rf"\*\*Q{q}\. .*?\*\*(.*?)\n\n"
        # pattern = rf"(?:\*\*Q{q}\. .+?\*\*|Q{q}\. .+?\n)(.*?)(?=\n\n|\Z)"
        # pattern = rf"(?:\*\*Q{q}\. .+?\*\*|Q{q}\. .+?\n)([\s\S]+?)(?=\n\n|\Z)"
        # pattern = rf"(?:\*\*Q{q}\. .+?\*\*|Q{q}\. .+?)([\s\S]+?)(?=\n\n|\Z)"
        pattern = rf"(?:\*\*Q{q}\. .+?\*\*|Q{q}\. .+?)(?:\s*\n)?([\s\S]+?)(?=\n\n|\Z)"
        match = re.search(pattern, response, re.DOTALL)
        answers[f"q{q}"] = match.group(1).strip() if match else ""
        return answers

    def categorise(self, response, q):
        """Categorise responses to question 1.
        
        Args:
            response (str): response.
        
        Returns:
            str: categorisation.
        """
        # response = response.lower()
        
        if q == "q1":
            if "Yes," in response or "Yes." in response:
                return "Yes"
            elif "No," in response or "No." in response:
                return "No"
            else:
                return "Other"
        elif q == "q2-av":
            # return brand
            brand_av = None
            model_av = None
            year_av = None
            # Cleanup of formatting
            response = re.sub(r"Autonomous Vehicle:|" +
                              r"Vehicle 1 \(Autonomous Vehicle\):|" +
                              r"Vehicle 1 \(Automated Vehicle\):|" +
                              r"The automated vehicle was a|" +
                              r"The autonomous vehicle was a|" +
                              r"Vehicle 1:", "Automated Vehicle:", response)
            response = re.sub(r"\*\*Automated Vehicle:\*\*", "Automated Vehicle:", response)
            response = re.sub(r"Apple Inc.", "Apple", response)
            response = re.sub(r"\(Not Specified\)", "Unknown", response)
            response = re.sub(r" \(indicated by a blank space\)", "", response)
            response = re.sub(r"Year:", "Year", response)
            response = re.sub(r"Brand:", "Brand", response)
            response = re.sub(r"Model:", "Model", response)
            # Manual filtering for specific types of road users
            if "Google AV" in response or "Google Auto LLC" in response or "Google Automated Vehicle" in response or "Google LLC" in response:  # noqa: E501
                brand_av = "Google"
            elif "Cruise AV" in response or "Cruise Vehicle" in response or "Cruise Automated Vehicle" in response or "Cruise LLC" in response:  # noqa: E501
                brand_av = "Cruise"
            elif "Zoom AV" in response:
                brand_av = "Zoom"
            # Year, brand, model
            # av_match_1 = re.search(r"Automated Vehicle: Year [^\n]*?(\d{4})?[^\n]*?, Brand ([A-Za-z]+), Model ([A-Za-z0-9\s]+)", response)  # noqa: E501
            av_match_1 = re.search(r"Automated Vehicle: Year (\d{4}), Brand ([A-Za-z-]+), Model ([A-Za-z0-9\s]+)", response)  # noqa: E501
            if not brand_av and av_match_1:
                # print(1, av_match_1, av_match_1.group(2))
                year_av = av_match_1.group(1)
                brand_av = av_match_1.group(2).strip()
                model_av = av_match_1.group(3).strip()
            # Year, brand_av, model
            av_match_2 = re.search(r"Automated Vehicle: (\d{4}), Brand ([A-Za-z-]+), Model ([A-Za-z0-9\s]+)", response)  # noqa: E501
            if not brand_av and av_match_2:
                # print(2, av_match_2, av_match_2.group(2))
                year_av = av_match_2.group(1) if av_match_2.group(1) else ""
                brand_av = av_match_2.group(2).strip()
                model_av = av_match_2.group(3).strip()
            # Year, brand_av, model
            av_match_3 = re.search(r"Automated Vehicle: Year (\d{4}), Brand ([A-Za-z\s]+)\.", response)  # noqa: E501
            if not brand_av and av_match_3:
                # print(3, av_match_3, av_match_3.group(2))
                year_av = av_match_3.group(1) if av_match_3.group(1) else ""
                brand_av = av_match_3.group(2).strip()
                # model_av = av_match_3.group(3).strip()
            # Other
            av_match_4 = re.search(r"Automated Vehicle:\s*(\d{4})?\s*([^.,*()]+)", response)
            if not brand_av and av_match_4:
                # print(4, av_match_4, av_match_4.group(2))
                year_av = av_match_4.group(1)
                brand_av = av_match_4.group(2).strip()
            # Cleanup brand
            if brand_av:
                if "waymo" in brand_av.lower() or "wayne" in brand_av.lower():
                    brand_av = "Waymo"
                elif "google" in brand_av.lower():
                    brand_av = "Google"
                elif "tesla" in brand_av.lower():
                    if "model X" in brand_av.lower():
                        model_av = "Model X"
                    elif "model S" in brand_av.lower():
                        model_av = "Model S"
                    elif "model 3" in brand_av.lower():
                        model_av = "Model 3"
                    brand_av = "Tesla"
                elif "cruise" in brand_av.lower():
                    brand_av = "Cruise"
                elif "apple" in brand_av.lower():
                    brand_av = "Apple"
                elif "ford" in brand_av.lower():
                    brand_av = "Ford"
                elif "toyota" in brand_av.lower() or "lexus" in brand_av.lower():
                    brand_av = "Toyota"
                elif "chrysler" in brand_av.lower():
                    brand_av = "Chrysler"
                elif "nissan" in brand_av.lower():
                    brand_av = "Nissan"
                elif "volvo" in brand_av.lower():
                    brand_av = "Volvo"
                elif "hyundai" in brand_av.lower():
                    brand_av = "Hyundai"
                elif "kia" in brand_av.lower():
                    brand_av = "Kia"
                elif "bmw" in brand_av.lower():
                    brand_av = "BMW"
                elif "chevrolet" in brand_av.lower():
                    brand_av = "Chevrolet"
                elif "land rover" in brand_av.lower() or "range rover" in brand_av.lower():
                    brand_av = "Land Rover"
                elif "lincoln" in brand_av.lower():
                    brand_av = "Lincoln"
                elif "honda" in brand_av.lower():
                    brand_av = "Honda"
                elif "subaru" in brand_av.lower():
                    brand_av = "Subaru"
                elif "nio" in brand_av.lower():
                    brand_av = "Nio"
                elif "mosaic" in brand_av.lower():
                    brand_av = "Mosaic"
                elif "mercedes-benz" in brand_av.lower() or "daimler" in brand_av.lower():
                    brand_av = "Mercedes-Benz"
                elif "Year" in brand_av:  # year fetched instead
                    brand_av = None
            # Brand not detected
            if not brand_av:
                # No match found
                brand_av = "Unknown"
                logger.debug(f"q2-av: no brand found for {response}.")
            # Model not detected
            if not model_av:
                # No match found
                model_av = "Unknown"
                # logger.debug(f"q2-av: no model found for {response}.")
            # YeR not detected
            if not year_av:
                # No match found
                year_av = "Unknown"
                # logger.debug(f"q2-av: no model found for {response}.")
            # Return fetched values
            return [brand_av, model_av, year_av]
        elif q == "q2-other_road_user":
            # Cleanup of formatting of answer: replace different formats of introducing other road user
            response = re.sub(r"\*\*Other Involved Road User:\*\*|" + 
                              r"\*\*Pedestrian:\*\*|" +
                              r"Other Involved Party:|", "Other Road User:", response)
            # Extract Other Involved Road User
            ru_match = re.search(r"Other Road User: \s*([^\n]*)", response)  # noqa: E501
            # Manual filtering for specific types of road users
            if ru_match and ("pedestrian" in ru_match.group(1).lower() or "walking" in ru_match.group(1).lower() or "male" in ru_match.group(1).lower() or "female" in ru_match.group(1).lower()):  # noqa: E501
                return "Pedestrian"
            elif ru_match and ("bicyclist" in ru_match.group(1).lower() or "cyclist" in ru_match.group(1).lower()):
                return "Cyclist"
            elif ru_match and ("scooter" in ru_match.group(1).lower() or "moped" in ru_match.group(1).lower()):
                return "Scooter"
            elif ru_match and ("fixed object" in ru_match.group(1).lower() or "stationary" in ru_match.group(1).lower()):  # noqa: E501
                return "Fixed object"
            elif ru_match and "driver" in ru_match.group(1).lower():
                return "Driver in vehicle"
            if ru_match:
                return ru_match.group(1).strip()
            # No match found
            logger.debug(f"q2-other_road_user: no match found for {response}.")
            return "unknown"
        elif q == "q2-other_vehicle":  
            # Cleanup of formatting of answer: replace different formats of introducing other car
            response = re.sub(r"\*\*Vehicle 2:\*\*|" + 
                              r"\*\*Other Car:\*\*", "Other Vehicle:", response)
            # Alternative format extraction
            ov_match = re.search(r"Other Vehicle: \s*(\d{4})?\s*([A-Za-z]+)\s+([A-Za-z0-9\s]+)\.", response)  # noqa: E501
            if ov_match:
                # year = ov_match.group(1)
                brand = ov_match.group(2).strip()
                model = ov_match.group(3).strip()
                # return f"{year} {brand} {model}".strip()
                return f"{brand} {model}".strip()
            else:
                # No match found
                logger.debug(f"q2-other_vehicle: no match found for {response}.")
        elif q == "q4":
            # Extract time and environmental conditions information
            weather = None
            lighting = None
            road_surface = None
            road_conditions = None
            
            # Extract weather conditions
            weather_match = re.search(r'\*\*Weather:\*\*\s*([^*\n.]+)', response)
            if weather_match:
                weather = weather_match.group(1).strip()
            
            # Extract lighting conditions
            lighting_match = re.search(r'\*\*Lighting Conditions:\*\*\s*([^*\n.]+)', response)
            if lighting_match:
                lighting = lighting_match.group(1).strip()
            
            # Extract road surface
            surface_match = re.search(r'\*\*Road Surface:\*\*\s*([^*\n.]+)', response)
            if surface_match:
                road_surface = surface_match.group(1).strip()
            
            # Extract road conditions
            conditions_match = re.search(r'\*\*Road Conditions:\*\*\s*([^*\n.]+)', response)
            if conditions_match:
                road_conditions = conditions_match.group(1).strip()
            
            return {
                'weather': weather,
                'lighting_conditions': lighting,
                'road_surface': road_surface,
                'road_conditions': road_conditions
            }

        elif q == "q5":
            # Define damage and collision categories for pattern matching
            damage_categories = {
                'minor': ['minor', 'slight', 'minimal', 'cosmetic'],
                'moderate': ['moderate', 'considerable', 'visible', 'damaged', 'dent'],
                'severe': ['severe', 'major', 'extensive', 'significant', 'heavy'],
                'total': ['total', 'destroyed', 'totaled']
            }
            
            # Extract collision type
            collision_type = None
            collision_match = re.search(r'\*\*Type of Collision:\*\*\s*([^*\n.]+)', response)
            if collision_match:
                collision_type = collision_match.group(1).strip()
            
            # Extract vehicle damage
            av_damage = None
            other_vehicle_damage = None
            vehicle_damage_match = re.search(r'\*\*Vehicle Damage:\*\*\s*([^*]+)', response)
            if vehicle_damage_match:
                vehicle_damage = vehicle_damage_match.group(1).strip()
                
                # Look for autonomous vehicle damage
                av_damage_match = re.search(r'(Tesla|AV|automated vehicle|autonomous vehicle)[^.]*(damage[^.]*)', vehicle_damage, re.IGNORECASE)
                if av_damage_match:
                    av_damage = av_damage_match.group(2).strip()
                
                # Look for other vehicle damage
                other_damage_match = re.search(r'([^T]oyota|Honda|Ford|Chrysler|other vehicle)[^.]*(damage[^.]*)', vehicle_damage, re.IGNORECASE)
                if other_damage_match:
                    other_vehicle_damage = other_damage_match.group(2).strip()
            
            # Classify damage severity for AV
            av_damage_category = "unknown"
            if av_damage:
                for category, keywords in damage_categories.items():
                    if any(keyword in av_damage.lower() for keyword in keywords):
                        av_damage_category = category
                        break
            
            # Extract injuries information
            injuries = None
            injuries_match = re.search(r'\*\*Injuries/Deaths/Property Damage:\*\*\s*([^*]+)', response)
            if injuries_match:
                injuries_text = injuries_match.group(1).strip()
                
                # Determine if there were injuries
                if re.search(r'injur(y|ies|ed)', injuries_text, re.IGNORECASE) and not re.search(r'no injur(y|ies|ed)', injuries_text, re.IGNORECASE):
                    injuries = True
                elif 'no injuries' in injuries_text.lower():
                    injuries = False
            
            return {
                'collision_type': collision_type,
                'av_damage_category': av_damage_category,
                'av_damage_description': av_damage,
                'other_vehicle_damage': other_vehicle_damage,
                'injuries': injuries
            }

        elif q == "q6":
            # Extract if AV is at fault
            av_at_fault = None
            if re.search(r'(autonomous|automated|AV).*?\bat fault\b', response, re.IGNORECASE):
                av_at_fault = True
            elif re.search(r'(pedestrian|other road user|driver).*?\bat fault\b', response, re.IGNORECASE):
                av_at_fault = False
            
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
            
            return {
                'av_at_fault': av_at_fault,
                'contributing_factors': contributing_factors
            }

        elif q == "q7":
            # Extract traffic conditions
            traffic_conditions = None
            traffic_match = re.search(r'\*\*Traffic:\*\*\s*([^*\n.]+)', response)
            if traffic_match:
                traffic_conditions = traffic_match.group(1).strip()
            
            # Extract vehicle movements
            av_movement = None
            other_movement = None
            same_direction = None
            same_lane = None
            
            # AV movement
            av_movement_match = re.search(r'(autonomous|automated|AV).*?(traveling|moving|driving|stopped)([^.]*)', response, re.IGNORECASE)
            if av_movement_match:
                av_movement = (av_movement_match.group(2) + av_movement_match.group(3)).strip()
            
            # Other road user movement
            other_movement_match = re.search(r'(pedestrian|other road user|driver).*?(walking|running|crossing|stopped|traveling|moving)([^.]*)', response, re.IGNORECASE)
            if other_movement_match:
                other_movement = (other_movement_match.group(2) + other_movement_match.group(3)).strip()
            
            # Check if same direction
            same_direction_match = re.search(r'(same|different) direction', response, re.IGNORECASE)
            if same_direction_match:
                same_direction = 'same' in same_direction_match.group(1).lower()
            
            # Check if same lane
            same_lane_match = re.search(r'(same|different) lanes?', response, re.IGNORECASE)
            if same_lane_match:
                same_lane = 'same' in same_lane_match.group(1).lower()
            
            return {
                'traffic_conditions': traffic_conditions,
                'av_movement': av_movement,
                'other_road_user_movement': other_movement,
                'same_direction': same_direction,
                'same_lane': same_lane
            }
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
        df["q1"] = df["response"].apply(lambda x: self.extract_answers(str(x), 1)["q1"])
        df["q1_category"] = df["q1"].apply(lambda x: self.categorise(str(x), "q1"))
        # Q2
        df["q2"] = df["response"].apply(lambda x: self.extract_answers(str(x), 2)["q2"])
        df[["q2_av_brand", "q2_av_model", "q2_av_year"]] = df["q2"].apply(
            lambda x: pd.Series(self.categorise(str(x), "q2-av"))
        )

        df["q2_other_road_user"] = df["q2"].apply(lambda x: self.categorise(str(x), "q2-other_road_user"))
        df["q2_other_vehicle"] = df["q2"].apply(lambda x: self.categorise(str(x), "q2-other_vehicle"))
        # Q3
        df["q3"] = df["response"].apply(lambda x: self.extract_answers(str(x), 3)["q3"])
        df["q3_category"] = df["q3"].apply(lambda x: self.categorise(str(x), "q3"))
        # Q4
        df["q4"] = df["response"].apply(lambda x: self.extract_answers(str(x), 4)["q4"])
        df["q4_category"] = df["q4"].apply(lambda x: self.categorise(str(x), "q4"))
        # Q5
        df["q5"] = df["response"].apply(lambda x: self.extract_answers(str(x), 5)["q5"])
        df["q5_category"] = df["q5"].apply(lambda x: self.categorise(str(x), "q5"))
        # Q6
        df["q6"] = df["response"].apply(lambda x: self.extract_answers(str(x), 6)["q6"])
        df["q6_category"] = df["q6"].apply(lambda x: self.categorise(str(x), "q6"))
        # Q7
        df["q7"] = df["response"].apply(lambda x: self.extract_answers(str(x), 7)["q7"])
        df["q7_category"] = df["q7"].apply(lambda x: self.categorise(str(x), "q7"))

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
