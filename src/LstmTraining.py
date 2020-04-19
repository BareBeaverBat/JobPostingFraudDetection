import re


def getNonEmptyLines(filePath):
    textLines = []
    with open(filePath) as dataFile:
        for dataLine in dataFile:
            if not re.match(r"^\s*$", dataLine):
                textLines.append(dataLine)

    return textLines

