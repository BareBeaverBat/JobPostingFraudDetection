import os
import pathlib



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


#dataset text attributes' summary files' names

TITLES_SUMMARY_FILENAME = "all_titles.txt"
LOCATIONS_SUMMARY_FILENAME = "all_locations.txt"
DEPARTMENTS_SUMMARY_FILENAME = "all_departments.txt"
COMPANY_PROFILES_SUMMARY_FILENAME = "all_company_profiles.txt"
DESCRIPTIONS_SUMMARY_FILENAME = "all_descriptions.txt"
REQUIREMENTS_SUMMARY_FILENAME = "all_requirements.txt"
BENEFITS_SUMMARY_FILENAME = "all_benefits.txt"




srcDirStr = os.getcwd()
srcDir = pathlib.Path(srcDirStr)
projectDir = srcDir.parent

DATA_PATH = os.path.join(projectDir, "data")
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
if not os.path.exists(RAW_DATA_PATH):
    os.mkdir(RAW_DATA_PATH)

PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "processed")
if not os.path.exists(PROCESSED_DATA_PATH):
    os.mkdir(PROCESSED_DATA_PATH)

PROCESSED_FILE_PREFIX = "cleaned_"

datasetDirName = PROCESSED_FILE_PREFIX + "kaggle_fake_job_postings"
datasetDirPath = os.path.join(PROCESSED_DATA_PATH, datasetDirName)

rawFname = "fake_job_postings.csv"
rawFpath = os.path.join(RAW_DATA_PATH, rawFname)

cleanedDataPath = os.path.join(datasetDirPath, PROCESSED_FILE_PREFIX + rawFname)




TRAIN_DATA_PATH = os.path.join(datasetDirPath, "train_data.csv")
VALIDATION_DATA_PATH = os.path.join(datasetDirPath, "valid_data.csv")
TEST_DATA_PATH = os.path.join(datasetDirPath, "test_data.csv")