import streamlit as st
import wikipedia
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

def lemma_me(sent):
    sentence_tokens = nltk.word_tokenize(sent.lower())
    pos_tags = nltk.pos_tag(sentence_tokens)

    sentence_lemmas = []
    for token, pos_tag in zip(sentence_tokens, pos_tags):
        if pos_tag[1][0].lower() in ['n', 'v', 'a', 'r']:
            lemma = lemmatizer.lemmatize(token, pos_tag[1][0].lower())
            sentence_lemmas.append(lemma)

    return sentence_lemmas

def process_question(text, question):
    sentence_tokens = nltk.sent_tokenize(text)
    sentence_tokens.append(question)

    tv = TfidfVectorizer(tokenizer=lemma_me)
    tf = tv.fit_transform(sentence_tokens)
    values = cosine_similarity(tf[-1], tf[:-1])
    index = values.argmax()
    coeff = values.max()

    if coeff > 0.3:
        return sentence_tokens[index]
    else:
        return None

def main():
    st.title("ChatBotOso")
    
    st.sidebar.title("Options")
    ask_checkbox = st.sidebar.checkbox("Ask Question")

    if ask_checkbox:
        try:
            topic = st.text_input("Please enter a topic:")
            text = wikipedia.page(topic).content
        except wikipedia.exceptions.DisambiguationError as e:
            st.write("Please choose a more specific topic from the options:")
            for option in e.options:
                st.write(option)
            return
        except wikipedia.exceptions.PageError:
            st.write("Topic not found.")
            return
        except wikipedia.exceptions.WikipediaException as e:
            #st.warning("An error occurred while retrieving information from Wikipedia.")
            return
        
        question = st.text_input("What do you want to know?")
        if st.button("Ask"):
            output = process_question(text, question)
            if output:
                st.write("Answer:", output)
            else:
                st.write("Sorry, I don't know.")

if __name__ == "__main__":
    main()