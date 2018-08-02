from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel

from nltk.stem.porter import *
from spacy.lang.en.stop_words import STOP_WORDS
from operator import add

from pyspark.ml.classification import NaiveBayes, NaiveBayesModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

yea = '\n\n\n`````````````````````START``````````````````````````\n'
yea1 = '\n\n\n`````````````````````END``````````````````````````\n'

categories = {
	'business':0.0,
	'politics':1.0,
	'science':2.0,
	'sports':3.0
}

stemmer = PorterStemmer()

def getCategory(title):
	category = title.split('/')[-2]
	return categories.get(category)

def cleanSentences(line):
	stemmedWords = []
	line = line.encode('ascii', 'ignore').decode('ascii')
	line = line.lower()
	words = line.split(' ')
	words = [word.strip() for word in words if len(word.strip()) > 0]
	words = [word for word in words if word not in STOP_WORDS]
	stemmedWords = [stemmer.stem(word) for word in words]
	return ' '.join(stemmedWords)


conf = SparkConf().setAppName('tfidf')
conf = conf.setMaster("local[*]")

sc   = SparkContext(conf=conf)
sqlContext = SQLContext(sc)


schema = StructType([StructField('label', FloatType(), True), \
					 StructField('sentences', StringType(), True)])

tokenizer = Tokenizer(inputCol="sentences", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=500)
idf = IDF(inputCol="rawFeatures",outputCol="features")


documents = sc.wholeTextFiles('../lab3_data/train_test/*')
documents = \
	documents.map(
		lambda (title, text): (
			title.encode('ascii', 'ignore').decode('ascii'),
			text.encode('ascii', 'ignore').decode('ascii')
		)
	)

documents = documents.map(lambda (title, text): (getCategory(title), cleanSentences(text)))

documentsDF = sqlContext.createDataFrame(documents, schema)

wordsData = tokenizer.transform(documentsDF)

featurizedData = hashingTF.transform(wordsData)

#idf
idfModel=idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

rescaledData =  rescaledData.select("label", "features")

training, test = rescaledData.randomSplit([0.8, 0.2], 1234)

#Logistic Regression
# Create a LogisticRegression instance. This instance is an Estimator.
lr = LogisticRegression(maxIter=20, regParam=0.01)

print("%s Training Started"+yea)

# Learn a LogisticRegression model. This uses the parameters stored in lr.
model1 = lr.fit(training)

prediction1 = model1.transform(test)
prediction1.show()

predictionAndLabel = prediction1.select('label', 'prediction')
predictionAndLabel = predictionAndLabel.rdd
accuracy11 = 100.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / test.count()

model1.save('./model/logisticRegModel6')


# Learn a Naive Bayes classifier.
# create the trainer and set its parameters
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# train the model
nb_model = nb.fit(training)

# select example rows to display.
predictions1 = nb_model.transform(test)
predictions1.show()

nb_model.save('./model/naiveBayesModel6')

# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="accuracy")
accuracy12 = evaluator.evaluate(predictions1)

#print("Test set accuracy = " + str(accuracy))


################Unknown data###########################


#Outputs of both the classifiers on test data:
print '%s Logistic Regression accuracy on Test data: %.4f %s' % (yea, accuracy11, yea1)

# cm = metrics.confusion_matrix(prediction1.select('label'),prediction1.select('prediction'))
# print(cm)

print '%s Naive Bayes Classifier accuracy on Test data: %s %s' % (yea, accuracy12*100, yea1)


