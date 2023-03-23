import pandas as pd
import os
import pandas as pd

def get_data_frame(filepath, is_csv, is_xls, is_excel):
    if is_csv:
        df = pd.read_csv(filepath, engine="python", error_bad_lines=False)
    elif is_xls or is_excel:
        # TODO needs to fix this later (setup parameter engine)
        engine = 'openpyxl' if is_excel else 'xlrd'
        df = pd.read_excel(filepath, engine=engine)
    return df


def get_missing_percentage(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    percent_missing = percent_missing.to_dict()
    return percent_missing


def get_sheet_names(filepath):
    xl = pd.ExcelFile(filepath)
    return xl.sheet_names
