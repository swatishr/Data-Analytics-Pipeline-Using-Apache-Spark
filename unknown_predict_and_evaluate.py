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



################Unknown data###########################
model1 = LogisticRegressionModel.load('./model/logisticRegModel5')
nb_model = NaiveBayesModel.load('./model/naiveBayesModel5')

documents1 = sc.wholeTextFiles('../lab3_data/unknown_data/*')
documents1 = \
	documents1.map(
		lambda (title, text): (
			title.encode('ascii', 'ignore').decode('ascii'),
			text.encode('ascii', 'ignore').decode('ascii')
		)
	)

documents1 = documents1.map(lambda (title, text): (getCategory(title), cleanSentences(text)))

documentsDF = sqlContext.createDataFrame(documents1, schema)

wordsData = tokenizer.transform(documentsDF)

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=500)
featurizedData = hashingTF.transform(wordsData)

#idf
idf = IDF(inputCol="rawFeatures",outputCol="features")
idfModel=idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

rescaledData =  rescaledData.select("label", "features")

prediction = model1.transform(rescaledData)
prediction.show()

predictionAndLabel = prediction.select('label', 'prediction')
predictionAndLabel = predictionAndLabel.rdd
accuracy = 100.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / rescaledData.count()

# select example rows to display.
predictions = nb_model.transform(rescaledData)
predictions.show()

# compute accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="accuracy")
accuracy1 = evaluator.evaluate(predictions)


#Outputs of both the classifiers on unknown data:
print '%s Logistic Regression accuracy unknown_data : %.4f %s' % (yea, accuracy, yea1)

# cm = metrics.confusion_matrix(prediction.select('label'),prediction.select('prediction'))
# print(cm)

print '%s Naive Bayes Classifier accuracy unknown_data: %s %s' % (yea, accuracy1*100, yea1)

# cm = metrics.confusion_matrix(predictions.select('label'),predictions.select('prediction'))
# print(cm)