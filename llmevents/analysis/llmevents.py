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

    def extract_answers(self, response):
        """Extract answers to each question from response text."""
        answers = {}
        for i in range(1, llme.common.get_configs('num_q') + 1):
            pattern = rf"\*\*Q{i}\. .*?\*\*(.*?)\n\n"
            match = re.search(pattern, response, re.DOTALL)
            answers[f"q{i}"] = match.group(1).strip() if match else ""
        return answers

    def categorise_response(self, response):
        """Categorise responses based on key terms."""
        response = response.lower()
        
        if "yes," in response or "yes." in response:
            return "yes"
        elif "no," in response or "no." in response:
            return "no"
        elif "collision" in response or "crash" in response or "accident" in response:
            return "collision/crash"
        elif "pedestrian" in response or "cyclist" in response:
            return "involving pedestrian/cyclist"
        elif "software issue" in response or "hardware issue" in response or "sensor failure" in response:
            return "technical failure"
        elif "human driver" in response or "manual intervention" in response:
            return "human intervention required"
        elif "near miss" in response or "close call" in response:
            return "near miss"
        elif "no issue" in response or "normal operation" in response:
            return "no incident"
        else:
            return "uncategorised"

    def process_answers(self, df):
        """Apply categorisation to each of the questions in the query."""
        num_q = llme.common.get_configs('num_q')
        logger.info(f'Processing output from {num_q} questions.')
        
        for i in range(1, num_q + 1):
            question_col = f"q{i}"
            if question_col not in df.columns:
                df[question_col] = df["response"].apply(lambda x: self.extract_answers(str(x))[question_col])
            df[f"q{i}_category"] = df[question_col].apply(lambda x: self.categorise_response(str(x)))
        
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
