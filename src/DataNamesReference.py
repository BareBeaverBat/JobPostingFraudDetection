import os
import pathlib
import numpy as np


# Dataframe labels

TITLE_LABEL = "title"
LOCATION_LABEL = "location"
DEPARTMENT_LABEL = "department"
COMPANY_PROFILE_LABEL = "company_profile"
DESCRIPTION_LABEL = "description"
REQUIREMENTS_LABEL = "requirements"
BENEFITS_LABEL = "benefits"

MIN_SALARY_LABEL = "min_salary"
MAX_SALARY_LABEL = "max_salary"
SALARY_RANGE_LABEL = "salary_range"
SALARY_MIDPT_LABEL = "salary_midpt"

EMPLOYMENT_TYPE_LABEL = "employment_type"
REQUIRED_EXPERIENCE_LABEL = "required_experience"
REQUIRED_EDUCATION_LABEL = "required_education"
INDUSTRY_LABEL = "industry"
FUNCTION_LABEL = "function"

TELECOMMUTING_LABEL = "telecommuting"
HAS_LOGO_LABEL = "has_logo"
HAS_QUESTIONS_LABEL = "has_questions"

FRAUDULENT_LABEL = "IS_FRAUDULENT"

COLUMN_DATA_TYPES = {TITLE_LABEL : np.unicode, LOCATION_LABEL : np.unicode, DEPARTMENT_LABEL : np.unicode,
                     COMPANY_PROFILE_LABEL : np.unicode, DESCRIPTION_LABEL : np.unicode, REQUIREMENTS_LABEL: np.unicode,
                     BENEFITS_LABEL : np.unicode, MIN_SALARY_LABEL : float, MAX_SALARY_LABEL : float,
                     SALARY_RANGE_LABEL : float, SALARY_MIDPT_LABEL : float, EMPLOYMENT_TYPE_LABEL : int,
                     REQUIRED_EXPERIENCE_LABEL : int, REQUIRED_EDUCATION_LABEL : int, INDUSTRY_LABEL : int,
                     FUNCTION_LABEL : int, TELECOMMUTING_LABEL : bool, HAS_LOGO_LABEL : bool, HAS_QUESTIONS_LABEL: bool,
                     FRAUDULENT_LABEL : bool}





#dataset text attributes' summary files' names

TITLES_SUMMARY_FILENAME = "all_titles.txt"
LOCATIONS_SUMMARY_FILENAME = "all_locations.txt"
DEPARTMENTS_SUMMARY_FILENAME = "all_departments.txt"
COMPANY_PROFILES_SUMMARY_FILENAME = "all_company_profiles.txt"
DESCRIPTIONS_SUMMARY_FILENAME = "all_descriptions.txt"
REQUIREMENTS_SUMMARY_FILENAME = "all_requirements.txt"
BENEFITS_SUMMARY_FILENAME = "all_benefits.txt"


#file names/paths

srcDirStr = os.getcwd()
srcDir = pathlib.Path(srcDirStr)
projectDir = srcDir.parent

dataPath = os.path.join(projectDir, "data")
if not os.path.exists(dataPath):
    os.mkdir(dataPath)

rawDataPath = os.path.join(dataPath, "raw")
if not os.path.exists(rawDataPath):
    os.mkdir(rawDataPath)

processedDataPath = os.path.join(dataPath, "processed")
if not os.path.exists(processedDataPath):
    os.mkdir(processedDataPath)

PROCESSED_FILE_PREFIX = "cleaned_"

datasetDirName = PROCESSED_FILE_PREFIX + "kaggle_fake_job_postings"
datasetDirPath = os.path.join(processedDataPath, datasetDirName)

rawFname = "fake_job_postings.csv"
rawFpath = os.path.join(rawDataPath, rawFname)

cleanedDataPath = os.path.join(datasetDirPath, PROCESSED_FILE_PREFIX + rawFname)



titlesSummaryFilePath = os.path.join(datasetDirPath, TITLES_SUMMARY_FILENAME)
locationsSummaryFilePath = os.path.join(datasetDirPath, LOCATIONS_SUMMARY_FILENAME)
departmentsSummaryFilePath = os.path.join(datasetDirPath, DEPARTMENTS_SUMMARY_FILENAME)
companyProfilesSummaryFilePath = os.path.join(datasetDirPath, COMPANY_PROFILES_SUMMARY_FILENAME)
descriptionsSummaryFilePath = os.path.join(datasetDirPath, DESCRIPTIONS_SUMMARY_FILENAME)
requirementsSummaryFilePath = os.path.join(datasetDirPath, REQUIREMENTS_SUMMARY_FILENAME)
benefitsSummaryFilePath = os.path.join(datasetDirPath, BENEFITS_SUMMARY_FILENAME)

EMPLOYMENT_TYPE_OPTIONS_FILE_NAME = "employment_type_options.txt"
REQUIRED_EXPERIENCE_OPTIONS_FILE_NAME= "required_experience_options.txt"
REQUIRED_EDUCATION_OPTIONS_FILE_NAME = "required_education_options.txt"
INDUSTRY_OPTIONS_FILE_NAME = "industry_options.txt"
FUNCTION_OPTIONS_FILE_NAME = "function_options.txt"


SPELL_CHECKER_PERSONAL_WORD_LIST_PATH= "SpellCheckerPersonalWordList.txt"

LANG_CODE = "en_US"


trainDataPath = os.path.join(datasetDirPath, "train_data.csv")
validationDataPath = os.path.join(datasetDirPath, "valid_data.csv")
testDataPath = os.path.join(datasetDirPath, "test_data.csv")


checkpointDirPath= os.path.join(projectDir, "checkpoints")
if not os.path.exists(checkpointDirPath):
    os.mkdir(checkpointDirPath)

tensorboardDirPath= os.path.join(projectDir, "tboard_logs")
if not os.path.exists(tensorboardDirPath):
    os.mkdir(tensorboardDirPath)


CSV_READ_ARGS = {"keep_default_na":False, "index_col":0, "dtype":COLUMN_DATA_TYPES}

