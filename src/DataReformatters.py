
from sklearn import preprocessing as skpreproc
import tensorflow.keras.preprocessing as preproc
from UtilityFuncs import getNonEmptyLines
import numpy as np
from Hyperparameters import *
from DataNamesReference import *

#todo implement use of word2vec pretrained embedding matrix

#do final preprocessing on each text attrib
def convertTitlesToPaddedSequences(dataDf):
    allTitles = getNonEmptyLines(TITLES_SUMMARY_FILE_PATH)
    allTitles.append(START_TOKEN)
    titleTokenizer = preproc.text.Tokenizer(num_words=TITLE_VOCAB_SIZE)
    titleTokenizer.fit_on_texts(allTitles)

    trainTitles = START_TOKEN + " " + dataDf[TITLE_LABEL].astype(str)
    trainTitleSequences = titleTokenizer.texts_to_sequences(trainTitles)
    paddedTrainTitleSequences = preproc.sequence.pad_sequences(trainTitleSequences, maxlen=MAX_TITLE_LEN,
                                                               padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)

    return paddedTrainTitleSequences

def convertLocationsToPaddedSequences(dataDf):
    allLocations = getNonEmptyLines(LOCATIONS_SUMMARY_FILE_PATH)
    allLocations.append(START_TOKEN)
    locationTokenizer = preproc.text.Tokenizer(num_words=LOCATION_VOCAB_SIZE)
    locationTokenizer.fit_on_texts(allLocations)

    trainLocations = START_TOKEN + " " + dataDf[LOCATION_LABEL].astype(str)
    trainLocationSequences = locationTokenizer.texts_to_sequences(trainLocations)
    paddedTrainLocationSequences = preproc.sequence.pad_sequences(trainLocationSequences, maxlen=MAX_LOCATION_LEN,
                                                               padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)

    return paddedTrainLocationSequences

def convertDepartmentsToPaddedSequences(dataDf):
    allDepartments = getNonEmptyLines(DEPARTMENTS_SUMMARY_FILE_PATH)
    allDepartments.append(START_TOKEN)
    departmentTokenizer = preproc.text.Tokenizer(num_words=DEPARTMENT_VOCAB_SIZE)
    departmentTokenizer.fit_on_texts(allDepartments)

    trainDepartments = START_TOKEN + " " +  dataDf[DEPARTMENT_LABEL].astype(str)
    trainDepartmentSequences = departmentTokenizer.texts_to_sequences(trainDepartments)
    paddedTrainDepartmentSequences = preproc.sequence.pad_sequences(trainDepartmentSequences, maxlen=MAX_DEPARTMENT_LEN,
                                                               padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)

    return paddedTrainDepartmentSequences

def convertCompanyProfilesToPaddedSequences(dataDf):
    allCompanyProfiles = getNonEmptyLines(COMPANY_PROFILES_SUMMARY_FILE_PATH)
    allCompanyProfiles.append(START_TOKEN)
    companyProfileTokenizer = preproc.text.Tokenizer(num_words=COMPANY_PROFILE_VOCAB_SIZE)
    companyProfileTokenizer.fit_on_texts(allCompanyProfiles)

    trainCompanyProfiles = START_TOKEN + " " + dataDf[COMPANY_PROFILE_LABEL].astype(str)
    trainCompanyProfileSequences = companyProfileTokenizer.texts_to_sequences(trainCompanyProfiles)
    paddedTrainCompanyProfileSequences = preproc.sequence.pad_sequences(trainCompanyProfileSequences, maxlen=MAX_COMPANY_PROFILE_LEN,
                                                               padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)

    return paddedTrainCompanyProfileSequences

def convertDescriptionsToPaddedSequences(dataDf):
    allDescriptions = getNonEmptyLines(DESCRIPTIONS_SUMMARY_FILE_PATH)
    allDescriptions.append(START_TOKEN)
    descriptionTokenizer = preproc.text.Tokenizer(num_words=DESCRIPTION_VOCAB_SIZE)
    descriptionTokenizer.fit_on_texts(allDescriptions)

    trainDescriptions = START_TOKEN + " " + dataDf[DESCRIPTION_LABEL].astype(str)
    trainDescriptionSequences = descriptionTokenizer.texts_to_sequences(trainDescriptions)
    paddedTrainDescriptionSequences = preproc.sequence.pad_sequences(trainDescriptionSequences, maxlen=MAX_DESCRIPTION_LEN,
                                                               padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)

    return paddedTrainDescriptionSequences

def convertRequirementsToPaddedSequences(dataDf):
    allRequirements = getNonEmptyLines(REQUIREMENTS_SUMMARY_FILE_PATH)
    allRequirements.append(START_TOKEN)
    requirementsTokenizer = preproc.text.Tokenizer(num_words=REQUIREMENTS_VOCAB_SIZE)
    requirementsTokenizer.fit_on_texts(allRequirements)

    trainRequirements = START_TOKEN + " " + dataDf[REQUIREMENTS_LABEL].astype(str)
    trainRequirementsSequences = requirementsTokenizer.texts_to_sequences(trainRequirements)
    paddedTrainRequirementsSequences = preproc.sequence.pad_sequences(trainRequirementsSequences, maxlen=MAX_REQUIREMENTS_LEN,
                                                               padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)

    return paddedTrainRequirementsSequences

def convertBenefitsToPaddedSequences(dataDf):
    allBenefits = getNonEmptyLines(BENEFITS_SUMMARY_FILE_PATH)
    allBenefits.append(START_TOKEN)
    benefitsTokenizer = preproc.text.Tokenizer(num_words=BENEFITS_VOCAB_SIZE)
    benefitsTokenizer.fit_on_texts(allBenefits)

    trainBenefits = START_TOKEN + " " + dataDf[BENEFITS_LABEL].astype(str)
    trainBenefitsSequences = benefitsTokenizer.texts_to_sequences(trainBenefits)
    paddedTrainBenefitsSequences = preproc.sequence.pad_sequences(trainBenefitsSequences, maxlen=MAX_BENEFITS_LEN,
                                                               padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)

    return paddedTrainBenefitsSequences


def normalizeSalaryData(dataDf):
    #make indices be within a subset of the data rather than the whole dataset
    # so that a sentinel can be made which applies to that subset of the data
    dataDf = dataDf.reset_index()
    numCases = dataDf.shape[0]
    nullSalaryIndices = dataDf.index[dataDf[SALARY_MIDPT_LABEL] == -1].tolist()

    outlierSalaryIndices = dataDf.index[dataDf[SALARY_MIDPT_LABEL] > MAX_REAL_SALARY].tolist()
    for outlierIndex in outlierSalaryIndices:
        print("cutting outlier salary value(s) at index ", outlierIndex, " within a subset of the data\n",
              "min= ", dataDf.loc[outlierIndex, MIN_SALARY_LABEL], "; max= ", dataDf.loc[outlierIndex, MAX_SALARY_LABEL],
              "; midpoint= ", dataDf.loc[outlierIndex, SALARY_MIDPT_LABEL], "; range= ", dataDf.loc[outlierIndex, SALARY_RANGE_LABEL])



    salaryMissingSentinel = np.zeros((numCases, 1))
    salaryMissingSentinel[nullSalaryIndices] = 1
    salaryOutlierSentinel = np.zeros((numCases, 1))
    salaryOutlierSentinel[outlierSalaryIndices] =1


    salaryMins = dataDf[MIN_SALARY_LABEL].to_numpy()
    salaryMins = np.reshape(salaryMins, (numCases, 1))
    salaryMins[nullSalaryIndices, 0] = np.nan
    salaryMins[outlierSalaryIndices, 0] = np.nan

    salaryMaxes = dataDf[MAX_SALARY_LABEL].to_numpy()
    salaryMaxes = np.reshape(salaryMaxes, (numCases, 1))
    salaryMaxes[nullSalaryIndices, 0] = np.nan
    salaryMaxes[outlierSalaryIndices, 0] = np.nan

    salaryMidpts = dataDf[SALARY_MIDPT_LABEL].to_numpy()
    salaryMidpts = np.reshape(salaryMidpts, (numCases, 1))
    salaryMidpts[nullSalaryIndices, 0] = np.nan
    salaryMidpts[outlierSalaryIndices, 0] = np.nan

    salaryRanges = dataDf[SALARY_RANGE_LABEL].to_numpy()
    salaryRanges = np.reshape(salaryRanges, (numCases, 1))
    salaryRanges[nullSalaryIndices, 0] = np.nan
    salaryRanges[outlierSalaryIndices, 0] = np.nan


    salaryData = np.concatenate((salaryMins, salaryMaxes, salaryMidpts, salaryRanges), axis=1)
    salaryData = skpreproc.scale(salaryData)

    salaryData[nullSalaryIndices, 0:4] = 0
    salaryData[outlierSalaryIndices, 0:4] = 0

    salaryData = np.concatenate((salaryMissingSentinel, salaryOutlierSentinel, salaryData), axis=1)

    return salaryData
