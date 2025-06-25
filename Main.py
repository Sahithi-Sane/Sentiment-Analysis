
import pandas as pd
from DataUtility import dataUtils, preProcessing, textVectorizer
dtUtils = dataUtils()
preProcess = preProcessing()
textVector = textVectorizer()
google_review_data = dtUtils.GetGoogleReviewData()

google_review_data = preProcess.removeEmptyReview(google_review_data)

preProcess.HandleTranslatedData(google_review_data)

preProcess.handlePunctuation(google_review_data)

preProcess.DoLowerCase(google_review_data)

preProcess.removeEmoji(google_review_data)

preProcess.handleURLAndHTMLAndSpecialCharacter(google_review_data)

preProcess.handleNumericValue(google_review_data)

preProcess.tokenizeText(google_review_data)

preProcess.handleStopWords(google_review_data)

preProcess.handleLemmatizer(google_review_data)

preProcess.handleNegation(google_review_data)

preProcess.addSentiwordAnalysis(google_review_data)

preProcess.joinTextData(google_review_data)

tfidf_matrix = textVector.handleTFIDFVectorizer(google_review_data)

count_matrix = textVector.bagOfWords(google_review_data)

Spacy_Embeddings = textVector.spacyWordEmbeddings(google_review_data)

pytorch_word2vec_matrix = textVector.pytorchWord2VecEmbedding(google_review_data)

keras_word2vec_matrix = textVector.kerasWord2VecEmbedding(google_review_data)



