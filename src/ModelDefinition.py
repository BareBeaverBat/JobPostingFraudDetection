
from tensorflow.keras import layers
from tensorflow.keras import Model
from Hyperparameters import *
import tensorflow.keras.regularizers as regs



#no recurrent_dropout on lstm because need to use GPU

#todo change variable names to _Tensors rather than _Layers?


#Description LSTM
descriptionInputLayer = layers.Input(name="descriptionTextInput", shape=(MAX_DESCRIPTION_LEN,), dtype="int32")

descriptionEmbedLayer = layers.Embedding(
    name="descriptionEmbedding", embeddings_regularizer=regs.l2(BASE_TEXT_EMBED_LAMBDA),
    input_dim=DESCRIPTION_VOCAB_SIZE, output_dim=TEXT_EMBED_DIM, mask_zero=True,
    input_length=MAX_DESCRIPTION_LEN)(descriptionInputLayer)

descriptionLstmLayer = layers.LSTM(name= "descriptionLstm",units=LSTM_SIZE, dropout=BASE_LSTM_DROPOUT,
                                   kernel_regularizer=regs.l2(BASE_LSTM_LAMBDA))
bidirDescriptionLstmLayer = layers.Bidirectional(descriptionLstmLayer)(descriptionEmbedLayer)

descriptionDropoutLayer = layers.Dropout(name="descriptionDropout", rate=BASE_DENSE_DROPOUT)(bidirDescriptionLstmLayer)
descriptionBatchNormLayer = layers.BatchNormalization(name="descriptionBatchNormalization")(descriptionDropoutLayer)

#secondary model output to allow for better training of the description-specific lstm
descriptionSidePrediction = layers.Dense(1, name="descSidePred", activation="sigmoid",
                                         kernel_regularizer=regs.l2(BASE_DENSE_LAMBDA)) (descriptionBatchNormLayer)


#Title LSTM
titleInputLayer = layers.Input(name="titleTextInput", shape=(MAX_TITLE_LEN,), dtype="int32")

titleEmbedLayer = layers.Embedding(
    name="titleEmbedding", embeddings_regularizer=regs.l2(BASE_TEXT_EMBED_LAMBDA),
    input_dim=TITLE_VOCAB_SIZE, output_dim=TEXT_EMBED_DIM, mask_zero=True,
    input_length=MAX_TITLE_LEN)(titleInputLayer)

titleLstmLayer = layers.LSTM(name= "titleLstm",units=LSTM_SIZE, dropout=BASE_LSTM_DROPOUT,
                                   kernel_regularizer=regs.l2(BASE_LSTM_LAMBDA))(titleEmbedLayer)

titleDropoutLayer = layers.Dropout(name="titleDropout", rate=BASE_DENSE_DROPOUT)(titleLstmLayer)
titleBatchNormLayer = layers.BatchNormalization(name="titleBatchNormalization")(titleDropoutLayer)

#secondary model output to allow for better training of the title-specific lstm
titleSidePrediction = layers.Dense(1, name="titleSidePred", activation="sigmoid",
                                         kernel_regularizer=regs.l2(BASE_DENSE_LAMBDA)) (titleBatchNormLayer)


#Location LSTM
locationInputLayer = layers.Input(name="locationTextInput", shape=(MAX_LOCATION_LEN,), dtype="int32")

locationEmbedLayer = layers.Embedding(
    name="locationEmbedding", embeddings_regularizer=regs.l2(BASE_TEXT_EMBED_LAMBDA),
    input_dim=LOCATION_VOCAB_SIZE, output_dim=TEXT_EMBED_DIM, mask_zero=True,
    input_length=MAX_LOCATION_LEN)(locationInputLayer)

locationLstmLayer = layers.LSTM(name= "locationLstm",units=LSTM_SIZE, dropout=BASE_LSTM_DROPOUT,
                                   kernel_regularizer=regs.l2(BASE_LSTM_LAMBDA))(locationEmbedLayer)

locationDropoutLayer = layers.Dropout(name="locationDropout", rate=BASE_DENSE_DROPOUT)(locationLstmLayer)
locationBatchNormLayer = layers.BatchNormalization(name="locationBatchNormalization")(locationDropoutLayer)

#secondary model output to allow for better training of the location-specific lstm
locationSidePrediction = layers.Dense(1, name="locSidePred", activation="sigmoid",
                                         kernel_regularizer=regs.l2(BASE_DENSE_LAMBDA)) (locationBatchNormLayer)


#Department LSTM
departmentInputLayer = layers.Input(name="departmentTextInput", shape=(MAX_DEPARTMENT_LEN,), dtype="int32")

departmentEmbedLayer = layers.Embedding(
    name="departmentEmbedding", embeddings_regularizer=regs.l2(BASE_TEXT_EMBED_LAMBDA),
    input_dim=DEPARTMENT_VOCAB_SIZE, output_dim=TEXT_EMBED_DIM, mask_zero=True,
    input_length=MAX_DEPARTMENT_LEN)(departmentInputLayer)

departmentLstmLayer = layers.LSTM(name= "departmentLstm",units=LSTM_SIZE, dropout=BASE_LSTM_DROPOUT,
                                   kernel_regularizer=regs.l2(BASE_LSTM_LAMBDA))(departmentEmbedLayer)

departmentDropoutLayer = layers.Dropout(name="departmentDropout", rate=BASE_DENSE_DROPOUT)(departmentLstmLayer)
departmentBatchNormLayer = layers.BatchNormalization(name="departmentBatchNormalization")(departmentDropoutLayer)

#secondary model output to allow for better training of the department-specific lstm
departmentSidePrediction = layers.Dense(1, name="deptSidePred", activation="sigmoid",
                                         kernel_regularizer=regs.l2(BASE_DENSE_LAMBDA)) (departmentBatchNormLayer)

#Company Profile LSTM
companyProfileInputLayer = layers.Input(name="companyProfileTextInput", shape=(MAX_COMPANY_PROFILE_LEN,), dtype="int32")

companyProfileEmbedLayer = layers.Embedding(
    name="companyProfileEmbedding", embeddings_regularizer=regs.l2(BASE_TEXT_EMBED_LAMBDA),
    input_dim=COMPANY_PROFILE_VOCAB_SIZE, output_dim=TEXT_EMBED_DIM, mask_zero=True,
    input_length=MAX_COMPANY_PROFILE_LEN)(companyProfileInputLayer)

companyProfileLstmLayer = layers.LSTM(name= "companyProfileLstm",units=LSTM_SIZE, dropout=BASE_LSTM_DROPOUT,
                                   kernel_regularizer=regs.l2(BASE_LSTM_LAMBDA))
bidirCompanyProfileLstmLayer=  layers.Bidirectional(companyProfileLstmLayer)(companyProfileEmbedLayer)

companyProfileDropoutLayer = layers.Dropout(name="companyProfileDropout", rate=BASE_DENSE_DROPOUT)(bidirCompanyProfileLstmLayer)
companyProfileBatchNormLayer = layers.BatchNormalization(name="companyProfileBatchNormalization")(companyProfileDropoutLayer)

#secondary model output to allow for better training of the companyProfile-specific lstm
companyProfileSidePrediction = layers.Dense(1, name="compProfSidePred", activation="sigmoid",
                                         kernel_regularizer=regs.l2(BASE_DENSE_LAMBDA)) (companyProfileBatchNormLayer)

#Requirements LSTM
requirementsInputLayer = layers.Input(name="requirementsTextInput", shape=(MAX_REQUIREMENTS_LEN,), dtype="int32")

requirementsEmbedLayer = layers.Embedding(
    name="requirementsEmbedding", embeddings_regularizer=regs.l2(BASE_TEXT_EMBED_LAMBDA),
    input_dim=REQUIREMENTS_VOCAB_SIZE, output_dim=TEXT_EMBED_DIM, mask_zero=True,
    input_length=MAX_REQUIREMENTS_LEN)(requirementsInputLayer)

requirementsLstmLayer = layers.LSTM(name= "requirementsLstm",units=LSTM_SIZE, dropout=BASE_LSTM_DROPOUT,
                                   kernel_regularizer=regs.l2(BASE_LSTM_LAMBDA))
bidirRequirementsLstmLayer = layers.Bidirectional(requirementsLstmLayer)(requirementsEmbedLayer)

requirementsDropoutLayer = layers.Dropout(name="requirementsDropout", rate=BASE_DENSE_DROPOUT)(bidirRequirementsLstmLayer)
requirementsBatchNormLayer = layers.BatchNormalization(name="requirementsBatchNormalization")(requirementsDropoutLayer)

#secondary model output to allow for better training of the requirements-specific lstm
requirementsSidePrediction = layers.Dense(1, name="reqsSidePred", activation="sigmoid",
                                         kernel_regularizer=regs.l2(BASE_DENSE_LAMBDA)) (requirementsBatchNormLayer)


#Benefits LSTM
benefitsInputLayer = layers.Input(name="benefitsTextInput", shape=(MAX_BENEFITS_LEN,), dtype="int32")

benefitsEmbedLayer = layers.Embedding(
    name="benefitsEmbedding", embeddings_regularizer=regs.l2(BASE_TEXT_EMBED_LAMBDA),
    input_dim=BENEFITS_VOCAB_SIZE, output_dim=TEXT_EMBED_DIM, mask_zero=True,
    input_length=MAX_BENEFITS_LEN)(benefitsInputLayer)

benefitsLstmLayer = layers.LSTM(name= "benefitsLstm",units=LSTM_SIZE, dropout=BASE_LSTM_DROPOUT,
                                   kernel_regularizer=regs.l2(BASE_LSTM_LAMBDA))
bidirBenefitsLstmLayer = layers.Bidirectional(benefitsLstmLayer)(benefitsEmbedLayer)

benefitsDropoutLayer = layers.Dropout(name="benefitsDropout", rate=BASE_DENSE_DROPOUT)(bidirBenefitsLstmLayer)
benefitsBatchNormLayer = layers.BatchNormalization(name="benefitsBatchNormalization")(benefitsDropoutLayer)

#secondary model output to allow for better training of the benefits-specific lstm
benefitsSidePrediction = layers.Dense(1, name="benefitsSidePred", activation="sigmoid",
                                         kernel_regularizer=regs.l2(BASE_DENSE_LAMBDA)) (benefitsBatchNormLayer)









employmentTypeInputLayer = layers.Input(name="employmentTypeInput", shape=(NUM_EMPLOYMENT_TYPE_OPTIONS,), dtype="float32")

requiredExperienceInputLayer = layers.Input(name="requiredExperienceInput", shape=(NUM_REQUIRED_EXPERIENCE_OPTIONS,), dtype="float32")

requiredEducationInputLayer = layers.Input(name="requiredEducationInput", shape=(NUM_REQUIRED_EDUCATION_OPTIONS,), dtype="float32")

industryInputLayer = layers.Input(name="industryInput", shape=(NUM_INDUSTRY_OPTIONS,), dtype="float32")

functionInputLayer = layers.Input(name="functionInput", shape=(NUM_FUNCTION_OPTIONS,), dtype="float32")

telecommutingInputLayer = layers.Input(name="telecommutingInput", shape=(1,), dtype="float32")
hasLogoInputLayer = layers.Input(name="hasLogoInput", shape=(1,), dtype="float32")
hasQuestionsInputLayer = layers.Input(name="hasQuestionsInput", shape=(1,), dtype="float32")



salaryInputLayer = layers.Input(name="salaryInput", shape=(6,), dtype="float32")



fullMergeLayer = layers.Concatenate(
    name="fullMerge", axis=1)([
    descriptionBatchNormLayer, titleBatchNormLayer, locationBatchNormLayer, departmentBatchNormLayer,
    companyProfileBatchNormLayer, requirementsBatchNormLayer, benefitsBatchNormLayer,
    employmentTypeInputLayer, requiredExperienceInputLayer, requiredEducationInputLayer, industryInputLayer,
    functionInputLayer, telecommutingInputLayer, hasLogoInputLayer, hasQuestionsInputLayer, salaryInputLayer])

fullDenseLayer1 = layers.Dense(name="firstFullDense", kernel_regularizer=regs.l2(BASE_DENSE_LAMBDA),
                               units=int(3.5*BASE_DENSE_SIZE), activation=DENSE_ACTIVATION)(fullMergeLayer)
fullDropoutLayer1 = layers.Dropout(name="firstFullDropout", rate=BASE_DENSE_DROPOUT)(fullDenseLayer1)

fullDenseLayer2 = layers.Dense(name="secondFullDense", kernel_regularizer=regs.l2(BASE_DENSE_LAMBDA),
                               units=int(3*BASE_DENSE_SIZE), activation=DENSE_ACTIVATION)(fullDropoutLayer1)
fullDropoutLayer2 = layers.Dropout(name="secondFullDropout", rate=BASE_DENSE_DROPOUT)(fullDenseLayer2)

fullDenseLayer3 = layers.Dense(name="thirdFullDense", kernel_regularizer=regs.l2(BASE_DENSE_LAMBDA),
                               units=int(2.5*BASE_DENSE_SIZE), activation=DENSE_ACTIVATION)(fullDropoutLayer2) # fullDropoutLayer2)
fullDropoutLayer3= layers.Dropout(name="thirdFullDropout", rate=BASE_DENSE_DROPOUT)(fullDenseLayer3)

fullDenseLayer4 = layers.Dense(name="fourthFullDense", kernel_regularizer=regs.l2(BASE_DENSE_LAMBDA),
                               units=int(2*BASE_DENSE_SIZE), activation=DENSE_ACTIVATION)(fullDropoutLayer3) # fullDropoutLayer2)
fullDropoutLayer4= layers.Dropout(name="fourthFullDropout", rate=BASE_DENSE_DROPOUT)(fullDenseLayer4)

fullDenseLayer5 = layers.Dense(name="fifthFullDense", kernel_regularizer=regs.l2(BASE_DENSE_LAMBDA),
                               units=int(1.5*BASE_DENSE_SIZE), activation=DENSE_ACTIVATION)(fullDropoutLayer4) # fullDropoutLayer2)
fullDropoutLayer5= layers.Dropout(name="fifthFullDropout", rate=BASE_DENSE_DROPOUT)(fullDenseLayer5)

fullDenseLayer6 = layers.Dense(name="sixthFullDense", kernel_regularizer=regs.l2(BASE_DENSE_LAMBDA),
                               units=int(1*BASE_DENSE_SIZE), activation=DENSE_ACTIVATION)(fullDropoutLayer5) # fullDropoutLayer2)
fullDropoutLayer6= layers.Dropout(name="sixthFullDropout", rate=BASE_DENSE_DROPOUT)(fullDenseLayer6)

fullDenseLayer7 = layers.Dense(name="seventhFullDense", kernel_regularizer=regs.l2(BASE_DENSE_LAMBDA),
                               units=int(0.5*BASE_DENSE_SIZE), activation=DENSE_ACTIVATION)(fullDropoutLayer6) # fullDropoutLayer2)
fullDropoutLayer7= layers.Dropout(name="seventhFullDropout", rate=BASE_DENSE_DROPOUT)(fullDenseLayer7)


finalBatchNormLayer = layers.BatchNormalization(name="finalBatchNormalization")(fullDropoutLayer7)


finalPrediction = layers.Dense(1, name="finalPred", activation="sigmoid") (finalBatchNormLayer)






fraudModel = Model(inputs=[employmentTypeInputLayer, requiredExperienceInputLayer, requiredEducationInputLayer,
                           industryInputLayer, functionInputLayer, telecommutingInputLayer, hasLogoInputLayer,
                           hasQuestionsInputLayer, salaryInputLayer,
                           descriptionInputLayer, titleInputLayer, locationInputLayer, departmentInputLayer,
                           companyProfileInputLayer, requirementsInputLayer, benefitsInputLayer],
                   outputs=[finalPrediction, descriptionSidePrediction, titleSidePrediction, locationSidePrediction,
                            departmentSidePrediction,
                            companyProfileSidePrediction, requirementsSidePrediction, benefitsSidePrediction])

NUM_OUTPUTS= 8