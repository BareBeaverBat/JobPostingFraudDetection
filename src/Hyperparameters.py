

# the plus 1 is for the start token
MAX_TITLE_LEN = 8+1
MAX_LOCATION_LEN =5+1
MAX_DEPARTMENT_LEN = 3+1
MAX_COMPANY_PROFILE_LEN = 200+1
MAX_DESCRIPTION_LEN = 300+1
MAX_REQUIREMENTS_LEN=200+1
MAX_BENEFITS_LEN = 125+1

START_TOKEN= "<TEXT_STARTS_AFTER_THIS>"

PADDING_TYPE = "post"
TRUNCATING_TYPE = "post"

#not eliminating any of the options
# extra plus 1 is for the start token
TITLE_VOCAB_SIZE=4708 + 1 + 1
LOCATION_VOCAB_SIZE=2335+1 + 1
DEPARTMENT_VOCAB_SIZE=1060+1 + 1
COMPANY_PROFILE_VOCAB_SIZE=13527+1 + 1
DESCRIPTION_VOCAB_SIZE=33470+1 + 1
REQUIREMENTS_VOCAB_SIZE=25259+1 + 1
BENEFITS_VOCAB_SIZE=11717+1 + 1


NUM_EMPLOYMENT_TYPE_OPTIONS = 6
NUM_REQUIRED_EXPERIENCE_OPTIONS=8
NUM_REQUIRED_EDUCATION_OPTIONS =14
NUM_INDUSTRY_OPTIONS=132
NUM_FUNCTION_OPTIONS=38

#used to eliminate outliers
MAX_REAL_SALARY= 5_000_000



TEXT_EMBED_DIM = 300

LSTM_SIZE=500

BASE_LSTM_DROPOUT=0.075
BASE_DENSE_DROPOUT=0.075

BASE_TEXT_EMBED_LAMBDA=0.002
BASE_DENSE_LAMBDA= 0.002
BASE_LSTM_LAMBDA=0.002



BASE_DENSE_SIZE= 500

DENSE_ACTIVATION="elu"



NUM_EPOCHS=50

BATCH_SIZE = 32

LEARNING_RATE=0.001 # 0.003

EARLY_STOPPING_PATIENCE=10
EARLY_STOPPING_MONITOR="val_finalPred_auc"
EARLY_STOPPING_MODE="max"
