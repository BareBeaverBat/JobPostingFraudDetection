import csv
import re
import pandas as pd
import numpy as np
import os
import pathlib
import shutil

import nltk

try:
    from nltk.corpus import stopwords
    dummyStopwords = stopwords.words("english")
except Exception as e:
    print("handling exception when importing stopwords from nltk: ", str(e))
    nltk.download("stopwords")
    from nltk.corpus import stopwords

# CSV column indices

TITLE_INDEX = 1
LOCATION_INDEX = 2
DEPARTMENT_INDEX = 3
SALARY_INDEX = 4
COMPANY_PROFILE_INDEX = 5
DESCRIPTION_INDEX = 6
REQUIREMENTS_INDEX = 7
BENEFITS_INDEX = 8
TELECOMMUTING_INDEX = 9
HAS_LOGO_INDEX = 10
HAS_QUESTIONS_INDEX = 11
EMPLOYMENT_TYPE_INDEX = 12
REQUIRED_EXPERIENCE_INDEX = 13
REQUIRED_EDUCATION_INDEX = 14
INDUSTRY_INDEX = 15
FUNCTION_INDEX = 16
FRAUDULENT_INDEX = 17

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

# salary processing constants

SALARY_RANGE_REGEX = re.compile(r"^\d+-\d+$")
SALARY_VAL_REGEX = re.compile(r"^\d+$")

# text preprocessing hyperparameters

baselineStopwords = set(stopwords.words('english'))

#after lower-casing
URL_REGEX = re.compile(r"#url_[a-z0-9]*#")
URL_REPLACEMENT_TOKEN = " url "
EMAIL_REGEX= re.compile(r"#email_[a-z0-9]*#")
EMAIL_REPLACEMENT_TOKEN=" email "
PHONE_REGEX= re.compile(r"#phone_[a-z0-9]*#")
PHONE_REPLACEMENT_TOKEN=" phone "

COMMA_SEPARATED_NUM_REGEX = re.compile(r"(\d+),(\d+)")
COMMA_SEPARATED_NUM_REPLACEMENT = r" \1\2 "

NONALPHANUMERIC_REGEX = re.compile(r"[^A-Za-z0-9^,!.\/'+-=]")

NON_ALPHANUM_SPAM_REGEX= re.compile(r"([^A-Za-z0-9^]+\s*){2,}")

MULT_WHITESPACE_REGEX = re.compile(r"\s{2,}")


TITLES_SUMMARY_FILENAME = "all_titles.txt"
LOCATIONS_SUMMARY_FILENAME = "all_locations.txt"
DEPARTMENTS_SUMMARY_FILENAME = "all_departments.txt"
COMPANY_PROFILES_SUMMARY_FILENAME = "all_company_profiles.txt"
DESCRIPTIONS_SUMMARY_FILENAME = "all_descriptions.txt"
REQUIREMENTS_SUMMARY_FILENAME = "all_requirements.txt"
BENEFITS_SUMMARY_FILENAME = "all_benefits.txt"





def findOrAdd(givenList, givenVal, canAdd = True):
    ind = -1
    for i, currVal in enumerate(givenList):
        if currVal == givenVal:
            ind = i
            break
    if ind == -1 and canAdd:
        givenList.append(givenVal)
        ind = len(givenList) - 1

    return ind


# classes for handling summary data from the data loading/preprocessing
class CategoriesSummary:
    def __init__(self):
        self.employmentTypeVals = []
        self.requiredExperienceVals = []
        self.requiredEducationVals = []
        self.industryVals = []
        self.functionVals = []

    def findOrAddEmploymentType(self, currEmploymentType, shouldAddNew):
        return findOrAdd(self.employmentTypeVals, currEmploymentType, shouldAddNew)

    def findOrAddRequiredExperience(self, currRequiredExperience, shouldAddNew):
        return findOrAdd(self.requiredExperienceVals, currRequiredExperience, shouldAddNew)

    def findOrAddRequiredEducation(self, currRequiredEducation, shouldAddNew):
        return findOrAdd(self.requiredEducationVals, currRequiredEducation, shouldAddNew)

    def findOrAddIndustry(self, currIndustry, shouldAddNew):
        return findOrAdd(self.industryVals, currIndustry, shouldAddNew)

    def findOrAddFunction(self, currFunction, shouldAddNew):
        return findOrAdd(self.functionVals, currFunction, shouldAddNew)

    def saveToFiles(self, dirPath):
        employmentTypeOptionsFilePath = os.path.join(dirPath, "employment_type_options.txt")
        with open(employmentTypeOptionsFilePath, mode="w") as employmentTypeOptionsFile:
            for employmentTypeOption in self.employmentTypeVals:
                employmentTypeOptionsFile.write("%s\n" % employmentTypeOption)

        requiredExperienceOptionsFilePath = os.path.join(dirPath, "required_experience_options.txt")
        with open(requiredExperienceOptionsFilePath, mode="w") as requiredExperienceOptionsFile:
            for requiredExperienceOption in self.requiredExperienceVals:
                requiredExperienceOptionsFile.write("%s\n" % requiredExperienceOption)

        requiredEducationOptionsFilePath = os.path.join(dirPath, "required_education_options.txt")
        with open(requiredEducationOptionsFilePath, mode="w") as requiredEducationOptionsFile:
            for requiredEducationOption in self.requiredEducationVals:
                requiredEducationOptionsFile.write("%s\n" % requiredEducationOption)

        industryOptionsFilePath = os.path.join(dirPath, "industry_options.txt")
        with open(industryOptionsFilePath, mode="w") as industryOptionsFile:
            for industryOption in self.industryVals:
                industryOptionsFile.write("%s\n" % industryOption)

        functionOptionsFilePath = os.path.join(dirPath, "function_options.txt")
        with open(functionOptionsFilePath, mode="w") as functionOptionsFile:
            for functionOption in self.functionVals:
                functionOptionsFile.write("%s\n" % functionOption)


class TextAttributeSummaries:
    def __init__(self):
        self.cumulTitlesText = ""
        self.cumulLocationsText = ""
        self.cumulDepartmentsText = ""
        self.cumulCompanyProfilesText = ""
        self.cumulDescriptionsText = ""
        self.cumulRequirementsText = ""
        self.cumulBenefitsText = ""

    def addTitle(self, currTitle):
        self.cumulTitlesText += currTitle + "\n\n"

    def addLocation(self, currLocation):
        self.cumulLocationsText += currLocation + "\n\n"

    def addDepartment(self, currDepartment):
        self.cumulDepartmentsText += currDepartment + "\n\n"

    def addCompanyProfile(self, currCompanyProfile):
        self.cumulCompanyProfilesText += currCompanyProfile + "\n\n"

    def addDescription(self, currDescription):
        self.cumulDescriptionsText += currDescription + "\n\n"

    def addRequirements(self, currRequirements):
        self.cumulRequirementsText += currRequirements + "\n\n"

    def addBenefits(self, currBenefits):
        self.cumulBenefitsText += currBenefits + "\n\n"

    def saveToFile(self, dirPath):
        titlesSummaryFilePath = os.path.join(dirPath, TITLES_SUMMARY_FILENAME)
        with open(titlesSummaryFilePath, mode="w") as titlesSummaryFile:
            titlesSummaryFile.write(self.cumulTitlesText)
        locationsSummaryFilePath = os.path.join(dirPath, LOCATIONS_SUMMARY_FILENAME)
        with open(locationsSummaryFilePath, mode="w") as locationsSummaryFile:
            locationsSummaryFile.write(self.cumulLocationsText)
        departmentsSummaryFilePath = os.path.join(dirPath, DEPARTMENTS_SUMMARY_FILENAME)
        with open(departmentsSummaryFilePath, mode="w") as departmentsSummaryFile:
            departmentsSummaryFile.write(self.cumulDepartmentsText)
        companyProfilesSummaryFilePath = os.path.join(dirPath, COMPANY_PROFILES_SUMMARY_FILENAME)
        with open(companyProfilesSummaryFilePath, mode="w") as companyProfilesSummaryFile:
            companyProfilesSummaryFile.write(self.cumulCompanyProfilesText)
        descriptionsSummaryFilePath = os.path.join(dirPath, DESCRIPTIONS_SUMMARY_FILENAME)
        with open(descriptionsSummaryFilePath, mode="w") as descriptionsSummaryFile:
            descriptionsSummaryFile.write(self.cumulDescriptionsText)
        requirementsSummaryFilePath = os.path.join(dirPath, REQUIREMENTS_SUMMARY_FILENAME)
        with open(requirementsSummaryFilePath, mode="w") as requirementsSummaryFile:
            requirementsSummaryFile.write(self.cumulRequirementsText)
        benefitsSummaryFilePath = os.path.join(dirPath, BENEFITS_SUMMARY_FILENAME)
        with open(benefitsSummaryFilePath, mode="w") as benefitsSummaryFile:
            benefitsSummaryFile.write(self.cumulBenefitsText)


# preprocessing functions

# based on
#  https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings
def cleanText(rawText, stopwordsList=None, stemmer=None):
    processedText = rawText.lower()

    processedText = re.sub(URL_REGEX, URL_REPLACEMENT_TOKEN, processedText)
    processedText = re.sub(EMAIL_REGEX, EMAIL_REPLACEMENT_TOKEN, processedText)
    processedText = re.sub(PHONE_REGEX, PHONE_REPLACEMENT_TOKEN, processedText)
    processedText = re.sub(r"&amp;", " and ", processedText)
    processedText = re.sub(COMMA_SEPARATED_NUM_REGEX, COMMA_SEPARATED_NUM_REPLACEMENT, processedText)


    processedText = re.sub(NONALPHANUMERIC_REGEX, " ", processedText)

    processedText = re.sub(r"what's", "what is ", processedText)
    processedText = re.sub(r"'s", " ", processedText)
    processedText = re.sub(r"\'ve", " have ", processedText)
    processedText = re.sub(r"can't", "cannot ", processedText)
    processedText = re.sub(r"n't", " not ", processedText)
    processedText = re.sub(r"i'm", "i am ", processedText)
    processedText = re.sub(r"\'re", " are ", processedText)
    processedText = re.sub(r"\'d", " would ", processedText)
    processedText = re.sub(r"\'ll", " will ", processedText)
    processedText = re.sub(r",", " ", processedText)
    processedText = re.sub(r"(\D)\.(\D)", r"\1 \2", processedText)
    processedText = re.sub(r"!", " ! ", processedText)
    processedText = re.sub(r"/", " ", processedText)
    processedText = re.sub(r"\^", " ^ ", processedText)
    processedText = re.sub(r"\+", " + ", processedText)
    processedText = re.sub(r"-", " - ", processedText)
    processedText = re.sub(r"=", " = ", processedText)
    processedText = re.sub(r"'", " ", processedText)
    processedText = re.sub(r"(\d+)(k)", r"\g<1>000", processedText)
    processedText = re.sub(r":", " : ", processedText)
    processedText = re.sub(r" e ?g ", " eg ", processedText)
    processedText = re.sub(r" b g ", " bg ", processedText)
    processedText = re.sub(r" u s ", " american ", processedText)
    processedText = re.sub(r"\0s", "0", processedText)
    processedText = re.sub(r" 9 11 ", "911", processedText)
    processedText = re.sub(r"e - mail", "email", processedText)
    processedText = re.sub(r"j k", "jk", processedText)

    processedText = re.sub(NON_ALPHANUM_SPAM_REGEX, " ", processedText)

    processedText = re.sub(MULT_WHITESPACE_REGEX, " ", processedText)

    if stopwordsList is not None:
        textArr = processedText.split()
        textArr = [currWord for currWord in textArr if not currWord in stopwordsList]
        processedText = " ".join(textArr)

    if stemmer is not None:
        textArr = processedText.split()
        stemmedWords = [stemmer.stem(currWord) for currWord in textArr]
        processedText = " ".join(stemmedWords)

    return processedText


def processJobListing(rawDataRow, categorySummariesObj, textAttributeSummariesObj, shouldBuildSummaries=True):
    processedListing = {}

    # copy the boolean values
    processedListing[TELECOMMUTING_LABEL] = rawDataRow[TELECOMMUTING_INDEX]
    processedListing[HAS_LOGO_LABEL] = rawDataRow[HAS_LOGO_INDEX]
    processedListing[HAS_QUESTIONS_LABEL] = rawDataRow[HAS_QUESTIONS_INDEX]
    processedListing[FRAUDULENT_LABEL] = rawDataRow[FRAUDULENT_INDEX]

    # one-hot encode the categorical attributes
    currEmploymentType = rawDataRow[EMPLOYMENT_TYPE_INDEX]
    currEmploymentTypeInd = categorySummariesObj.findOrAddEmploymentType(currEmploymentType, shouldBuildSummaries)
    processedListing[EMPLOYMENT_TYPE_LABEL] = currEmploymentTypeInd

    currRequiredExperience = rawDataRow[REQUIRED_EXPERIENCE_INDEX]
    currRequiredExperienceInd = categorySummariesObj.findOrAddRequiredExperience(currRequiredExperience, shouldBuildSummaries)
    processedListing[REQUIRED_EXPERIENCE_LABEL] = currRequiredExperienceInd

    currRequiredEducation = rawDataRow[REQUIRED_EDUCATION_INDEX]
    currRequiredEducationInd = categorySummariesObj.findOrAddRequiredEducation(currRequiredEducation, shouldBuildSummaries)
    processedListing[REQUIRED_EDUCATION_LABEL] = currRequiredEducationInd

    currIndustry = rawDataRow[INDUSTRY_INDEX]
    currIndustryInd = categorySummariesObj.findOrAddIndustry(currIndustry, shouldBuildSummaries)
    processedListing[INDUSTRY_LABEL] = currIndustryInd

    currFunction = rawDataRow[FUNCTION_INDEX]
    currFunctionInd = categorySummariesObj.findOrAddFunction(currFunction, shouldBuildSummaries)
    processedListing[FUNCTION_LABEL] = currFunctionInd

    # process salary attribute, eliminating invalid salary entries
    currSalaryText = rawDataRow[SALARY_INDEX]

    minSalaryVal = -1
    maxSalaryVal = -1
    salaryRange = -1
    salaryMidpt = -1

    if SALARY_RANGE_REGEX.match(currSalaryText):
        salaryStrs = currSalaryText.split("-")
        minSalaryStr = salaryStrs[0]
        maxSalaryStr = salaryStrs[1]

        minSalaryVal = float(minSalaryStr)
        maxSalaryVal = float(maxSalaryStr)
        salaryRange = maxSalaryVal - minSalaryVal
        salaryMidpt = (maxSalaryVal + minSalaryVal) / 2
    elif SALARY_VAL_REGEX.match(currSalaryText):
        minSalaryVal = float(currSalaryText)
        maxSalaryVal = minSalaryVal
        salaryRange = 0
        salaryMidpt = minSalaryVal
    else:
        pass  # use default invalid values

    processedListing[MIN_SALARY_LABEL] = minSalaryVal
    processedListing[MAX_SALARY_LABEL] = maxSalaryVal
    processedListing[SALARY_RANGE_LABEL] = salaryRange
    processedListing[SALARY_MIDPT_LABEL] = salaryMidpt

    # basic processing of text attributes
    titleVal = rawDataRow[TITLE_INDEX]
    cleanedTitleVal = cleanText(titleVal)
    processedListing[TITLE_LABEL] = cleanedTitleVal
    if shouldBuildSummaries:
        textAttributeSummariesObj.addTitle(cleanedTitleVal)

    locationVal = rawDataRow[LOCATION_INDEX]
    cleanedLocationVal = cleanText(locationVal)
    processedListing[LOCATION_LABEL] = cleanedLocationVal
    if shouldBuildSummaries:
        textAttributeSummariesObj.addLocation(cleanedLocationVal)

    departmentVal = rawDataRow[DEPARTMENT_INDEX]
    cleanedDepartmentVal = cleanText(departmentVal)
    processedListing[DEPARTMENT_LABEL] = cleanedDepartmentVal
    if shouldBuildSummaries:
        textAttributeSummariesObj.addDepartment(cleanedDepartmentVal)

    companyProfileVal = rawDataRow[COMPANY_PROFILE_INDEX]
    cleanedCompanyProfileVal = cleanText(companyProfileVal, baselineStopwords)
    processedListing[COMPANY_PROFILE_LABEL] = cleanedCompanyProfileVal
    if shouldBuildSummaries:
        textAttributeSummariesObj.addCompanyProfile(cleanedCompanyProfileVal)

    descriptionVal = rawDataRow[DESCRIPTION_INDEX]
    cleanedDescriptionVal = cleanText(descriptionVal, baselineStopwords)
    processedListing[DESCRIPTION_LABEL] = cleanedDescriptionVal
    if shouldBuildSummaries:
        textAttributeSummariesObj.addDescription(cleanedDescriptionVal)

    requirementsVal = rawDataRow[REQUIREMENTS_INDEX]
    cleanedRequirementsVal = cleanText(requirementsVal, baselineStopwords)
    processedListing[REQUIREMENTS_LABEL] = cleanedRequirementsVal
    if shouldBuildSummaries:
        textAttributeSummariesObj.addRequirements(cleanedRequirementsVal)

    benefitsVal = rawDataRow[BENEFITS_INDEX]
    cleanedBenefitsVal = cleanText(benefitsVal, baselineStopwords)
    processedListing[BENEFITS_LABEL] = cleanedBenefitsVal
    if shouldBuildSummaries:
        textAttributeSummariesObj.addBenefits(cleanedBenefitsVal)

    return processedListing


def loadData(fpath):
    allCategories = CategoriesSummary()
    allTextAttributes = TextAttributeSummaries()
    processedData = []

    with open(fpath, encoding="utf-8") as raw_csv:
        dataReader = csv.reader(raw_csv)
        tempHeaderRow = next(dataReader)  # eliminate header row

        # preprocesses the data as it's loaded
        for row in dataReader:
            processedRow = processJobListing(row, allCategories, allTextAttributes)
            processedData.append(processedRow)

    return processedData, allCategories, allTextAttributes


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

if __name__ == "__main__":
    if os.path.exists(datasetDirPath):
        print("overwriting directory at path ", datasetDirPath)
        shutil.rmtree(datasetDirPath)
    os.mkdir(datasetDirPath)

    cleanedData, categorySummaries, textAttributeSummaries = loadData(rawFpath)
    cleanedDataDf = pd.DataFrame(cleanedData)

    dataSaveResult = cleanedDataDf.to_csv(cleanedDataPath)
    if dataSaveResult is not None:
        print("saving dataframe failed with a message (about csv format?): ", dataSaveResult)

    categorySummaries.saveToFiles(datasetDirPath)
    textAttributeSummaries.saveToFile(datasetDirPath)



