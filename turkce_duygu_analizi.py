# gerekli kütüphaneler
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, classification_report, precision_score
from sklearn.model_selection import cross_val_score


# yapılacak indirmeler 
"""nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')"""
stopWords = set(stopwords.words('turkish'))


# aldığımız metni temizlediğimiz, noktalama işaretleri ve gereksiz kelimelerden arındırdığımız fonksiyon
def pre_process(text):
    text = text.lower()
    text = re.sub("[^abcçdefgğhıijklmnoöprsştuüvyz]", " ", text)
    text = nltk.word_tokenize(text)
    text = [word for word in text if not word in set(stopwords.words("turkish"))]
    lemma = nltk.WordNetLemmatizer()
    text = [lemma.lemmatize(word) for word in text]
    text = " ".join(text)
    return text


# veriyi okuyup doğruluğunu kontrol edebileceğimiz alan
dftrain = pd.read_csv('data/train.csv', encoding='unicode_escape')
dftest = pd.read_csv('data/test.csv', encoding='unicode_escape')
#print(dftrain.head(5))
#print(dftest.head(5))


# işlenmiş veriyi yeni bir sütuna eklediğimiz alan
dftrain["smooth_text"] = dftrain["comment"].apply(lambda x : pre_process(x))
dftest["smooth_text"] = dftest["comment"].apply(lambda x : pre_process(x))


# regression için veriyi hazırladığımız alan
xtrain = dftrain["smooth_text"]
xtest = dftest["smooth_text"]
ytrain = dftrain["Label"]
ytest = dftest["Label"]
#print("xtest", xtest.shape,"xtrain",  xtrain.shape, "ytest", ytest.shape, "ytrain", ytrain.shape)


# logistic regression uygulayıp verimizi yakınsadığımız alan
LogisticRegression = Pipeline([('tfidf', TfidfVectorizer()),('clf', LogisticRegression())])
LogisticRegression .fit(xtrain, ytrain)


# model başarımızı değerlendirmek için çizeceğimiz matris 
def plot_confusion_matrix(ytestt, yprediction):
    conf_mat = confusion_matrix(ytestt, yprediction)
    #print(conf_mat)
    fig = plt.figure(figsize=(6,6))
    plt.matshow(conf_mat, cmap=plt.cm.Blues, fignum=1)
    plt.yticks(range(2), range(2))
    plt.xticks(range(2), range(2))
    plt.colorbar();
    for i in range(2):
        for j in range(2):
            plt.text(i-0.1,j+0.05, str(conf_mat[j, i]), color='tab:cyan')


# değerlerimizi hesapladığımız, yazdırdığımız ve matris çizdirdiğimiz alan
cv_scores = cross_val_score(LogisticRegression, xtrain, ytrain, cv=10)
#print("Average CV score: %.2f" % cv_scores.mean())
result = LogisticRegression.predict(xtest)
cr = classification_report(ytest, result)
#print(cr) 
#print('Train Accuracy : %.3f'%LogisticRegression.score(xtrain, ytrain))
#print('Test Accuracy : %.3f'%LogisticRegression.score(xtest, ytest))
ypred = LogisticRegression.predict(xtest)
#print("The precision score : ", precision_score(ytest, ypred ,average='macro'))
#print("The recall score : ", recall_score(ytest, ypred,average='macro'))
#print("The f1 score : ", f1_score(ytest, ypred ,average='macro'))

plot_confusion_matrix(ytest, LogisticRegression.predict(xtest))


# analizi yapılacak cümle için fonksiyon
def predict_sentiment(sentence):
    processed_sentence = pre_process(sentence)
    prediction = LogisticRegression.predict([processed_sentence])[0]
    probability = LogisticRegression.predict_proba([processed_sentence])
    return prediction, probability[0]


# kullanıcıdan cümleyi girdi olarak aldığımız alan
sentence = str(input("Cümleyi girin : "))
prediction, probability = predict_sentiment(sentence)


# cümlenin tahmini değerlerini yazdırdığımız ve datasete eklediğimiz alan
if prediction == 0:
    print("Girilen bu cümle NEGATİF!")
else:
    print("Girilen cümle POZİTİF!")
print("Olumlu olma ihtimali : %.3f, Olumsuz olma ihtimali : %.3f" % (probability[1], probability[0]))
