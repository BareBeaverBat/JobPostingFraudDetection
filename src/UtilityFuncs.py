import re
import matplotlib.pyplot as plt
import sklearn.metrics as sklMetrics
import numpy as np


def getNonEmptyLines(filePath):
    textLines = []
    with open(filePath) as dataFile:
        for dataLine in dataFile:
            if not re.match(r"^\s*$", dataLine):
                textLines.append(dataLine)

    return textLines


def evaluateModel(modelName, datasetType, labels, predictions):
    roundedPreds = np.round(predictions)
    accScore = sklMetrics.accuracy_score(labels, roundedPreds)
    balancedAccScore = sklMetrics.balanced_accuracy_score(labels, roundedPreds)
    f1Score = sklMetrics.f1_score(labels, roundedPreds)
    precision = sklMetrics.precision_score(labels, roundedPreds)
    recall = sklMetrics.recall_score(labels, roundedPreds)

    aurocScore = sklMetrics.roc_auc_score(labels, predictions)

    print("on ", datasetType, " data, the ", modelName, " achieved:\n",
          "accuracy= ", accScore, "; balanced accuracy = ", balancedAccScore,
          "\nF1 score= ", f1Score, "; AUROC score= ", aurocScore,
          "\nPrecision= ", precision, "; Recall= ", recall)

    falsePositiveRates, truePositiveRates, rocThresholds = sklMetrics.roc_curve(labels, predictions,
                                                                                drop_intermediate=False)

    summaryRocThresholds = np.quantile(rocThresholds, (0, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1))
    print("Summary of thresholds (of model output) for ROC curve\n",
          "(0th, 10th, 20th...80th, 90th, 100th percentiles):\n",
          summaryRocThresholds)



    plt.plot(falsePositiveRates, truePositiveRates, 'bo')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    rocPlotTitle= "ROC curve for " + modelName + " on " + datasetType + " data"
    plt.title(rocPlotTitle)
    plt.show()