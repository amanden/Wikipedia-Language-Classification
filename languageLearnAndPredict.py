
import math
import pandas as pd
import pickle
import sys

class DecisionTree:
    __slots__ = 'val', 'leftNode' , 'rightNode'

    def __init__(self, val, leftNode = None, rightNode=None):
        self.val = val
        self.leftNode = leftNode
        self.rightNode = rightNode

def setAttributes(line, goal):
    line = " "+line
    rows = [ False, False, False, False, False, False, False, False, False, False , False]
    if line.count(" de ") > 0:
        rows[0] = True

    if line.count(" van ") > 0:
        rows[1] = True

    if line.count(" een ") > 0:
        rows[2] = True

    if line.count("ij") > 0:
        rows[3] = True

    if line.count("aa") > 0:
        rows[4] = True

    if line.count(" and ") > 0:
        rows[5] = True

    if line.count(" the ") > 0:
        rows[6] = True

    if line.count(" of ") > 0:
        rows[7] = True

    if line.count(" to ") > 0:
        rows[8] = True

    sum = 0
    for words in line.split():
        sum = sum + len(words)
    avg = sum / 15
    if avg > 5:
        rows[9] = True

    rows[10] = goal
    return rows


def traindt(examplefile, hypothesisOut):
    sentences = open(examplefile , 'r')
    punctuations = "!()-[]{};:\,\”\“<>.?&"
    attributes = ["de", "van", "een", "ij", "aa", "and","the", "of", "to", "avg", "goal"]

    data = []
    ex = []
    for sentence in sentences:
        goal = sentence[:2]
        ex.append(sentence[:3])
        line = sentence[3:].lower()
        for p in punctuations:
            if line.count(p) > 0:
                line = line.replace(p , "")

        rows = setAttributes(line, goal)
        data.append(rows)

    df = pd.DataFrame(data, columns=attributes, index=ex)

    attributes.remove("goal")
    tree = buildDecisionTree(df, attributes, df)
    pickle.dump(tree, open(hypothesisOut, 'wb'))

def buildDecisionTree(df, attributes, parent_df):
    # print(gain(attributes, df))
    if df.shape[0] == 0:
        return DecisionTree(pluralityValue(parent_df))
    sameClass , lang = checkClassification(df)
    if sameClass == True:
        return DecisionTree(lang)
    if len(attributes) == 0:
        return DecisionTree(pluralityValue(df))

    attr = gain(attributes, df)

    leftFalsedf = df.loc[df[attr] == False]
    rightTruedf = df.loc[df[attr] == True]
    attributes.remove(attr)

    leftFalseTree = buildDecisionTree(leftFalsedf, attributes, df)
    rightTrueTree = buildDecisionTree(rightTruedf , attributes, df)

    # print("attr = ", attr, "left = ", leftFalseTree.val , "right = ", rightTrueTree.val)

    return DecisionTree(attr, leftFalseTree , rightTrueTree)


def pluralityValue(dataFrame ):
    nlCount = 0
    enCount = 0
    for row in range(dataFrame.shape[0]):
        goalVal = dataFrame["goal"][row]
        if goalVal == 'nl':
            nlCount += 1
        elif goalVal == 'en':
            enCount += 1
    if nlCount > enCount:
        return 'nl'
    else:
        return 'en'

def checkClassification(dataFrame):
    flagSameClass = False
    lang = 'en'
    nlCount = 0
    enCount = 0
    for row in range(dataFrame.shape[0]):
        goalVal = dataFrame["goal"][row]
        if goalVal == 'nl':
            nlCount += 1
        elif goalVal == 'en':
            enCount += 1
    if nlCount == 0 or enCount == 0:
        flagSameClass = True
    if nlCount > enCount:
        lang = 'nl'
    return  flagSameClass, lang


def gain(attributes, dataFrame):
    # print(dataFrame[attributes[10]][1])
    total = dataFrame.shape[0]

    nlCount = 0
    enCount = 0
    for row in range(total):
        val = dataFrame["goal"][row]
        if val == 'nl':
            nlCount = nlCount + 1
        if val == 'en':
            enCount = enCount + 1

    goalEntropy = entropy(nlCount, enCount)

    maxGain = 0
    maxGainAttr = ''
    for attr in attributes:
        attrGain = 0
        attrTrueCount = 0
        attrFalseCount = 0

        attrTrueNLcount = 0
        attrFalseNLCount = 0

        attrTrueENcount = 0
        attrFalseENcount = 0
        for row in range(total-1):
            attrVal = dataFrame[attr][row]
            goalVal = dataFrame["goal"][row]
            if attrVal == True:
                attrTrueCount = attrTrueCount + 1
                if goalVal == 'nl':
                    attrTrueNLcount += 1
                if goalVal == 'en':
                    attrTrueENcount += 1

            if attrVal == False:
                attrFalseCount = attrFalseCount + 1
                if goalVal == 'nl':
                    attrFalseNLCount += 1
                if goalVal == 'en':
                    attrFalseENcount += 1
        attrEntropy = ((attrTrueCount / total) * entropy(attrTrueNLcount, attrTrueENcount)) + ((attrFalseCount / total) * entropy(attrFalseNLCount, attrFalseENcount))
        attrGain = goalEntropy - attrEntropy
        if attrGain > maxGain:
            maxGain = attrGain
            maxGainAttr = attr
    return maxGainAttr

def entropy(a, b):
    # print("a ", a, " b ", b)
    if (a+b) > 0:
        pa = a / (a + b)
        pb = b / (a + b)
    else:
        return 0
    if pa == 0 or pa == 1 or pb == 0 or pb == 1:
        return 0
    entro = - (pa * math.log(pa, 2)) - (pb * math.log(pb, 2))
    return entro

def trainada(examplefile, hypothesisOut):
    sentences = open(examplefile , 'r')
    punctuations = "!()-[]{};:\,\”\“<>.?&"
    attributes = ["de", "van", "een", "ij", "aa", "and","the", "of", "to", "avg", "goal"]

    data = []
    for sentence in sentences:
        goal = sentence[:2]
        line = sentence[3:].lower()
        for p in punctuations:
            if line.count(p) > 0:
                line = line.replace(p , "")

        rows = setAttributes(line, goal)
        data.append(rows)

    df = pd.DataFrame(data, columns=attributes)
    hypothesis = buildAdaboost(df)
    # print(hypothesis)
    pickle.dump(hypothesis, open(hypothesisOut, 'wb'))

def buildAdaboost(dataFrame):

    K = 10
    weights = []
    N = dataFrame.shape[0]  #no. of rows
    # print(N)
    for i in range(N):
        weights.append( 1 / N )

    examples = ["de", "van", "een", "ij", "aa", "and","the", "of", "to", "avg"]

    # print(weights)
    hypothesis = []
    for k in range(K):
        stump = []
        minError = math.inf
        attrCorrectRows = []
        attr = ''
        for n in range(len(examples)):
            # print(examples[n],"---")
            error = 0
            correctRows = []
            for i in range(N):
                if checkCorrectness(i, examples[n], dataFrame) == True:
                    correctRows.append(i)
                else:
                    error = error + weights[i]
                if error < minError:
                    minError = error
                    attrCorrectRows = correctRows
                    attr = examples[n]

        for rows in attrCorrectRows:
            weights[rows] *= minError /(1 - minError + 0.0000001)

        totalWt = sum(weights)
        for i in range(N):
            weights[rows] = weights[rows] / totalWt

        attrWt = math.log((1 - minError) / (error + 0.000001), 2)
        stump.append(attr)
        stump.append(attrWt)

        hypothesis.append(stump)
        examples.remove(attr)
        examples.append(attr)

    return hypothesis

def predictAdaboost(hypothesis, file):
    testData = open(file, 'r')
    punctuations = "!()-[]{};:\,\”\“<>.?&"
    attributes = ["de", "van", "een", "ij", "aa", "and", "the", "of", "to", "avg", "goal"]
    langMap = {"de":"nl", "van":"nl", "een":"nl", "ij":"nl", "aa":"nl", "and":"en", "the":"en", "of":"en", "to":"en", "avg":"nl" }

    data = []
    for sentence in testData:
        goal = sentence[:2]
        line = sentence[3:].lower()
        for p in punctuations:
            if line.count(p) > 0:
                line = line.replace(p, "")

        rows = setAttributes(line, '')
        data.append(rows)
    df = pd.DataFrame(data, columns=attributes)
    # print(df)
    for row in range(df.shape[0]):
        sum = 0
        for attr in hypothesis:
            if langMap[attr[0]] == 'nl':
                if df[attr[0]][row] == True:
                    sum = sum + attr[1] * 1
                else:
                    sum = sum + attr[1] * -1
            else:
                if df[attr[0]][row] == False:
                    sum = sum + attr[1] * 1
                else:
                    sum = sum + attr[1] * -1
        if sum > 0:
            print("nl")
        else:
            print("en")


def checkCorrectness(row, attr, dataFrame):
    value = dataFrame[attr][row]
    goal = dataFrame["goal"][row]
    if attr == "de":
        if (value == True and goal == 'nl') or (value == False and goal == 'en'):
            return True
    if attr == "van" :
        if (value == True and goal == 'nl') or (value == False and goal == 'en'):
            return True
    if attr == "een" :
        if (value == True and goal == 'nl') or (value == False and goal == 'en'):
            return True
    if attr == "ij" :
        if (value == True and goal == 'nl') or (value == False and goal == 'en'):
            return True
    if attr == "aa" :
        if (value == True and goal == 'nl') or (value == False and goal == 'en'):
            return True
    if attr == "and" :
        if (value == True and goal == 'en') or (value == False and goal == 'nl'):
            return True
    if attr == "the" :
        if (value == True and goal == 'en') or (value == False and goal == 'nl'):
            return True
    if attr == "of" :
        if (value == True and goal == 'en') or (value == False and goal == 'nl'):
            return True
    if attr == "to" :
        if (value == True and goal == 'en') or (value == False and goal == 'nl'):
            return True
    if attr == "avg" :
        if (value == True and goal == 'nl') or (value == False and goal == 'en'):
            return True
    return False


def predictDT(decisionTree, file):
    # decisionTree = pickle.load(open(hypothesis, 'rb'))
    testData = open(file, 'r')

    punctuations = "!()-[]{};:\,\”\“<>.?&"
    attributes = ["de", "van", "een", "ij", "aa", "and", "the", "of", "to", "avg", "goal"]

    data = []
    for sentence in testData:
        goal = sentence[:2]
        # ex.append(sentence[:3])
        line = sentence[3:].lower()
        for p in punctuations:
            if line.count(p) > 0:
                line = line.replace(p, "")

        rows = setAttributes(line, '')
        data.append(rows)
    df = pd.DataFrame(data, columns=attributes)
    for row in range(df.shape[0]):
        value = predictSentence(df, decisionTree , row)
        print(value)


def predictSentence(dataFrame , decisionTree,  row):
    # print(dataFrame.shape[0])
    if decisionTree.leftNode == None or decisionTree.rightNode == None:
        return decisionTree.val
    if dataFrame[decisionTree.val][row] == False:
        return predictSentence(dataFrame, decisionTree.leftNode, row)
    else:
        return predictSentence(dataFrame, decisionTree.rightNode, row)

def predict(hypothesis, file):
    savedObject = pickle.load(open(hypothesis, 'rb'))
    if isinstance(savedObject , DecisionTree):
        predictDT(savedObject, file)
    elif isinstance(savedObject, list):
        predictAdaboost(savedObject, file)
    else:
        print("Wrong hypothesis file")

def main():

    type = sys.argv[1]
    if type == "train":
        learner = sys.argv[5]
        tdata = sys.argv[3]
        ohypo = sys.argv[4]
        if learner == 'dt':
            traindt(tdata, ohypo)
        else:
            trainada(tdata, ohypo)
    if type == "predict":
        hypo = sys.argv[2]
        tesdata = sys.argv[3]
        predict(hypo, tesdata)

if __name__ == '__main__':
    main()
