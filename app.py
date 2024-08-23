import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from design import Ui_MainWindow

import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#------------------------------------------------------------
class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        #------------------------------------------------------------
        # Load spaCy language model (you can replace 'en_core_web_sm' with a larger model)
        self.nlp = spacy.load('en_core_web_sm')
        #------------------------------------------------------------
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        #------------------------------------------------------------
        self.send()
        #------------------------------------------------------------
    
    def get_phrases(self):
        text = self.ui.phrases.toPlainText()
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        return lines
    
    def get_text(self):
        input_text = self.ui.phrases.toPlainText()
        # Tokenize and preprocess the input text
        doc = self.nlp(str(input_text))
        standardized_phrases = self.get_phrases()
        # Embed the standardized phrases and input text
        standardized_phrase_embeddings = [self.nlp(phrase).vector for phrase in standardized_phrases]
        input_text_embedding = np.mean([sentence.vector for sentence in doc.sents], axis=0)

        # Calculate cosine similarity scores
        similarity_scores = cosine_similarity([input_text_embedding], standardized_phrase_embeddings)

        # Find the most similar standardized phrase
        most_similar_index = np.argmax(similarity_scores)

        # Print the suggestion
        most_similar_phrase = standardized_phrases[most_similar_index]

        print("Recommended Replacement:", most_similar_phrase)
        print("Similarity Score:", similarity_scores[0][most_similar_index])
        results = f'"Recommended replacement: " {most_similar_phrase} ":  score = " {similarity_scores[0][most_similar_index]}'
        self.ui.restults.setText(results)

    def send(self):
        self.ui.btn_go.clicked.connect(self.get_text)




if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    sys.exit(app.exec_())
