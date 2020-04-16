import csv
import re
import pandas as pd
import numpy as np
import os
import pathlib
import shutil

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

baselineStopwords = []  # TODO

URL_REGEX = re.compile(r"#URL_[A-Za-z0-9]*#")
URL_REPLACEMENT_TOKEN = " <URL> "

NONALPHANUMERIC_REGEX = re.compile(r"[^A-Za-z0-9^,!.\/'+-=]")

MULT_WHITESPACE_REGEX = re.compile(r"\s{2,}")


def findOrAdd(givenList, givenVal):
    ind = -1
    for i, currVal in enumerate(givenList):
        if currVal == givenVal:
            ind = i
            break
    if ind == -1:
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

    def findOrAddEmploymentType(self, currEmploymentType):
        return findOrAdd(self.employmentTypeVals, currEmploymentType)

    def findOrAddRequiredExperience(self, currRequiredExperience):
        return findOrAdd(self.requiredExperienceVals, currRequiredExperience)

    def findOrAddRequiredEducation(self, currRequiredEducation):
        return findOrAdd(self.requiredEducationVals, currRequiredEducation)

    def findOrAddIndustry(self, currIndustry):
        return findOrAdd(self.industryVals, currIndustry)

    def findOrAddFunction(self, currFunction):
        return findOrAdd(self.functionVals, currFunction)

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
        titlesSummaryFilePath = os.path.join(dirPath, "all_titles.txt")
        with open(titlesSummaryFilePath, mode="w") as titlesSummaryFile:
            titlesSummaryFile.write(self.cumulTitlesText)
        locationsSummaryFilePath = os.path.join(dirPath, "all_locations.txt")
        with open(locationsSummaryFilePath, mode="w") as locationsSummaryFile:
            locationsSummaryFile.write(self.cumulLocationsText)
        departmentsSummaryFilePath = os.path.join(dirPath, "all_departments.txt")
        with open(departmentsSummaryFilePath, mode="w") as departmentsSummaryFile:
            departmentsSummaryFile.write(self.cumulDepartmentsText)
        companyProfilesSummaryFilePath = os.path.join(dirPath, "all_company_profiles.txt")
        with open(companyProfilesSummaryFilePath, mode="w") as companyProfilesSummaryFile:
            companyProfilesSummaryFile.write(self.cumulCompanyProfilesText)
        descriptionsSummaryFilePath = os.path.join(dirPath, "all_descriptions.txt")
        with open(descriptionsSummaryFilePath, mode="w") as descriptionsSummaryFile:
            descriptionsSummaryFile.write(self.cumulDescriptionsText)
        requirementsSummaryFilePath = os.path.join(dirPath, "all_requirements.txt")
        with open(requirementsSummaryFilePath, mode="w") as requirementsSummaryFile:
            requirementsSummaryFile.write(self.cumulRequirementsText)
        benefitsSummaryFilePath = os.path.join(dirPath, "all_benefits.txt")
        with open(benefitsSummaryFilePath, mode="w") as benefitsSummaryFile:
            benefitsSummaryFile.write(self.cumulBenefitsText)


# preprocessing functions

# based on
#  https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings
def cleanText(rawText, stopwordsList=None, stemmer=None):
    processedText = rawText.lower()

    processedText = re.sub(URL_REGEX, URL_REPLACEMENT_TOKEN, processedText)

    processedText = re.sub(r"&amp;", " and ", processedText)

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
    processedText = re.sub(r"\.", " ", processedText)
    processedText = re.sub(r"!", " ! ", processedText)
    processedText = re.sub(r"/", " ", processedText)
    processedText = re.sub(r"\^", " ^ ", processedText)
    processedText = re.sub(r"\+", " + ", processedText)
    processedText = re.sub(r"-", " - ", processedText)
    processedText = re.sub(r"=", " = ", processedText)
    processedText = re.sub(r"'", " ", processedText)
    processedText = re.sub(r"(\d+)(k)", r"\g<1>000", processedText)
    processedText = re.sub(r":", " : ", processedText)
    processedText = re.sub(r" e g ", " eg ", processedText)
    processedText = re.sub(r" b g ", " bg ", processedText)
    processedText = re.sub(r" u s ", " american ", processedText)
    processedText = re.sub(r"\0s", "0", processedText)
    processedText = re.sub(r" 9 11 ", "911", processedText)
    processedText = re.sub(r"e - mail", "email", processedText)
    processedText = re.sub(r"j k", "jk", processedText)

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


def processJobListing(rawDataRow, categorySummariesObj, textAttributeSummariesObj):
    processedListing = {}

    # copy the boolean values
    processedListing[TELECOMMUTING_LABEL] = rawDataRow[TELECOMMUTING_INDEX]
    processedListing[HAS_LOGO_LABEL] = rawDataRow[HAS_LOGO_INDEX]
    processedListing[HAS_QUESTIONS_LABEL] = rawDataRow[HAS_QUESTIONS_INDEX]
    processedListing[FRAUDULENT_LABEL] = rawDataRow[FRAUDULENT_INDEX]

    # one-hot encode the categorical attributes
    currEmploymentType = rawDataRow[EMPLOYMENT_TYPE_INDEX]
    currEmploymentTypeInd = categorySummariesObj.findOrAddEmploymentType(currEmploymentType)
    processedListing[EMPLOYMENT_TYPE_LABEL] = currEmploymentTypeInd

    currRequiredExperience = rawDataRow[REQUIRED_EXPERIENCE_INDEX]
    currRequiredExperienceInd = categorySummariesObj.findOrAddRequiredExperience(currRequiredExperience)
    processedListing[REQUIRED_EXPERIENCE_LABEL] = currRequiredExperienceInd

    currRequiredEducation = rawDataRow[REQUIRED_EDUCATION_INDEX]
    currRequiredEducationInd = categorySummariesObj.findOrAddRequiredEducation(currRequiredEducation)
    processedListing[REQUIRED_EDUCATION_LABEL] = currRequiredEducationInd

    currIndustry = rawDataRow[INDUSTRY_INDEX]
    currIndustryInd = categorySummariesObj.findOrAddIndustry(currIndustry)
    processedListing[INDUSTRY_LABEL] = currIndustryInd

    currFunction = rawDataRow[FUNCTION_INDEX]
    currFunctionInd = categorySummariesObj.findOrAddFunction(currFunction)
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
    cleanedTitleVal = cleanText(titleVal, baselineStopwords)
    processedListing[TITLE_LABEL] = cleanedTitleVal
    textAttributeSummariesObj.addTitle(cleanedTitleVal)

    locationVal = rawDataRow[LOCATION_INDEX]
    cleanedLocationVal = cleanText(locationVal, baselineStopwords)
    processedListing[LOCATION_LABEL] = cleanedLocationVal
    textAttributeSummariesObj.addLocation(cleanedLocationVal)

    departmentVal = rawDataRow[DEPARTMENT_INDEX]
    cleanedDepartmentVal = cleanText(departmentVal, baselineStopwords)
    processedListing[DEPARTMENT_LABEL] = cleanedDepartmentVal
    textAttributeSummariesObj.addDepartment(cleanedDepartmentVal)

    companyProfileVal = rawDataRow[COMPANY_PROFILE_INDEX]
    cleanedCompanyProfileVal = cleanText(companyProfileVal, baselineStopwords)
    processedListing[COMPANY_PROFILE_LABEL] = cleanedCompanyProfileVal
    textAttributeSummariesObj.addCompanyProfile(cleanedCompanyProfileVal)

    descriptionVal = rawDataRow[DESCRIPTION_INDEX]
    cleanedDescriptionVal = cleanText(descriptionVal, baselineStopwords)
    processedListing[DESCRIPTION_LABEL] = cleanedDescriptionVal
    textAttributeSummariesObj.addDescription(cleanedDescriptionVal)

    requirementsVal = rawDataRow[REQUIREMENTS_INDEX]
    cleanedRequirementsVal = cleanText(requirementsVal, baselineStopwords)
    processedListing[REQUIREMENTS_LABEL] = cleanedRequirementsVal
    textAttributeSummariesObj.addRequirements(cleanedRequirementsVal)

    benefitsVal = rawDataRow[BENEFITS_INDEX]
    cleanedBenefitsVal = cleanText(benefitsVal, baselineStopwords)
    processedListing[BENEFITS_LABEL] = cleanedBenefitsVal
    textAttributeSummariesObj.addBenefits(cleanedBenefitsVal)

    return processedListing


def loadData(fpath):
    allCategories = CategoriesSummary()
    allTextAttributes = TextAttributeSummaries()
    processedData = []

    with open(fpath, encoding="utf-8") as raw_csv:
        dataReader = csv.reader(raw_csv)
        tempHeaderRow = next(dataReader)  # eliminate header row
        tempRow1 = next(dataReader)
        tempRow2= next(dataReader)
        tempRow3 = next(dataReader)
        tempRow4 = next(dataReader)

        # preprocesses the data as it's loaded
        for row in dataReader:
            processedRow = processJobListing(row, allCategories, allTextAttributes)
            processedData.append(processedRow)
            # print(processedRow, flush=True)

    return processedData, allCategories, allTextAttributes


currDirStr = os.getcwd()
currDir = pathlib.Path(currDirStr)
projectDir = currDir.parent

DATA_PATH = os.path.join(projectDir, "data")
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "processed")

PROCESSED_FILE_PREFIX = "cleaned_"

datasetDirName = "kaggle_fake_job_postings"
datasetDirPath = os.path.join(PROCESSED_DATA_PATH, datasetDirName)
# print(list(datasetDirPath))
# print(datasetDirPath)
if os.path.exists(datasetDirPath):
    print("overwriting directory at path ", datasetDirPath)
    shutil.rmtree(datasetDirPath)
os.mkdir(datasetDirPath)

rawFname = "fake_job_postings.csv"
rawFpath = os.path.join(RAW_DATA_PATH, rawFname)

cleanedData, categorySummaries, textAttributeSummaries = loadData(rawFpath)
cleanedDataDf = pd.DataFrame(cleanedData)

cleanedDataPath = os.path.join(datasetDirPath, PROCESSED_FILE_PREFIX + rawFname)
dataSaveResult = cleanedDataDf.to_csv(cleanedDataPath)
if dataSaveResult is not None:
    print("saving dataframe failed with a message (about csv format?): ", dataSaveResult)

categorySummaries.saveToFiles(datasetDirPath)
textAttributeSummaries.saveToFile(datasetDirPath)



