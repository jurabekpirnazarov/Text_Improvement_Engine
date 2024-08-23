import streamlit as st
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load spaCy language model
nlp = spacy.load('en_core_web_sm')

# Streamlit app
st.title("Text Improvement Suggestion Tool")

# Input text section
st.write("### Input Text")
input_text = st.text_area("Enter your text here:")

# Standardized phrases section
st.write("### Standardized Phrases")
standardized_phrases = st.text_area("Enter standardized phrases (one per line):", height=200)

# Button to trigger suggestion
if st.button("Suggest Improvements"):
    # Convert standardized phrases to a list
    standardized_phrases = [phrase.strip() for phrase in standardized_phrases.split('\n') if phrase.strip()]

    # Check if input text and standardized phrases are provided
    if input_text and standardized_phrases:
        # Process the input text
        doc = nlp(input_text)
        input_text_embedding = np.mean([sentence.vector for sentence in doc.sents], axis=0)

        # Embed the standardized phrases
        standardized_phrase_embeddings = [nlp(phrase).vector for phrase in standardized_phrases]

        # Calculate cosine similarity scores
        similarity_scores = cosine_similarity([input_text_embedding], standardized_phrase_embeddings)

        # Find the most similar standardized phrase
        most_similar_index = np.argmax(similarity_scores)
        most_similar_phrase = standardized_phrases[most_similar_index]
        similarity_score = similarity_scores[0][most_similar_index]

        # Display the suggestion
        st.write("### Suggested Improvement")
        st.write(f"**Recommended Replacement:** {most_similar_phrase}")
        st.write(f"**Similarity Score:** {similarity_score:.2f}")
    else:
        st.write("Please enter both text and standardized phrases.")

# Run the Streamlit app
if __name__ == "__main__":
    st.write("Ready to make your text more professional!")
