# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import os
import pandas as pd
from tqdm import tqdm
from pdf2image import convert_from_path     # lz added 4/4/2024
import base64      # lz added 4/4/2024


import gptevents as gpte

# warning about partial assignment
pd.options.mode.chained_assignment = None  # default='warn'

logger = gpte.CustomLogger(__name__)  # use custom logger


def encode_image(image_path):
    with open(image_path, "rb") as imageFile:
        return base64.b64encode(imageFile.read()).decode('utf-8')


class ChatGPT:
    # pandas dataframe with extracted data
    data = pd.DataFrame()
    save_p = False  # save data as pickle file
    load_p = False  # load data as pickle file
    save_csv = False  # save data as csv file
    # pickle file for saving data
    file_p = 'data.p'
    # csv file for saving data
    file_data_csv = 'data.csv'

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

    def read_data(self, filter_data=True, clean_data=True, analyse_data=True):
        """Read data into an attribute.

        Args:
            filter_data (bool, optional): flag for filtering data.
            clean_data (bool, optional): clean data.
            analyse_data (bool, optional): analyse data.

        Returns:
            dataframe: updated dataframe.
        """
        # load data
        if self.load_p:
            df = gpte.common.load_from_p(self.file_p, 'chatgpt data')
        # get data based on the reports
        else:
            # pandas df to store data
            df = pd.DataFrame(columns=('report', 'response'))
            # df = df.transpose()
            # go over all reports
            for file in tqdm(os.listdir(self.files_reports)):
                # get pages as base64_image strings
                pages = self.pdf_to_base64_image(file)
                # feed all pages in the report to GPT-4V at once
                df = pd.concat([df, self.process_gptv(file, pages)], ignore_index=True)
            # clean data
            if clean_data:
                df = self.clean_data(df)
            # filter data
            if filter_data:
                df = self.filter_data(df)
            # analyse data
            if analyse_data:
                df = self.analyse_data(df)
            # sort columns alphabetically
            df = df.reindex(sorted(df.columns), axis=1)
            # report people that attempted study
            logger.info('Processed {} reports.', df.shape[0])
        # save to pickle
        if self.save_p:
            gpte.common.save_to_p(self.file_p, df, 'chatgpt data')
        # save to csv
        if self.save_csv:
            df.to_csv(os.path.join(gpte.settings.output_dir, self.file_data_csv),
                      index=False)
            logger.info('Saved data to csv file {}',
                        self.file_data_csv + '.csv')
        # update attribute
        self.data = df
        # return df with data
        return df

    def pdf_to_base64_image(self, file):
        """Turn pages of the PDF file with the report to base64 strings.
        Args:
            file (str): Name of file of the report.

        Returns:
            base64_image (list): List of pages as base64 strings.
        """
        # create full path of the file with the report
        file = os.fsdecode(file)
        logger.info('Turning report {} into base64_image.', file)
        file = os.path.join(self.files_reports, file)
        # f = open(file, 'r')
        # each page is 1 base64_image
        base64_images = []

        # TODO: turn pdfs into base64 (LZ)
        imgs = convert_from_path(file)
        temp_jpg = "output_images_jpg"
        if not os.path.exists(temp_jpg):
            os.makedirs(temp_jpg)
        for i, image in enumerate(imgs):
            # save generated images. This can be overwritten.
            image_path = os.path.join(temp_jpg, f"page_{i+1}.jpg")
            image.save(image_path, 'JPEG')
            base64_images.append(encode_image(image_path))
        # close image
        # f.close()
        return base64_images

    # TODO: call for GPT4-V (PB)
    def process_gptv(self, file, pages):
        """Receive responses from GPT4-V for all pages at once.
        Args:
            file (str): File with report.
            pages (list): List of pages as base64 strings.

        Returns:
            dataframe: dataframe with responses.
        """
        logger.error('Not implemented.')
        data = {'report': [file], 'response': ['todo']}
        df = pd.DataFrame(data)
        print(df)
        return df

    # TODO: call for GPT4-V (PB)
    def analyse_data(self, df):
        """Analyse responses from GPT4-V for all reports.
        Args:
            df (TYPE): Dataframe with responses from GPT4-V for all reports.

        Returns:
            dataframe: updated dataframe.
        """
        logger.error('Analysis of data not implemented.')
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
