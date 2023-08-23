import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import wikipedia

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

def process(text, question):
    sentence_tokens = nltk.sent_tokenize(text)
    sentence_tokens.append(question)

    tv = TfidfVectorizer(tokenizer=lemma_me)
    tf = tv.fit_transform(sentence_tokens)
    values = cosine_similarity(tf[-1], tf)
    index = values.argsort()[0][-2]
    values_flat = values.flatten()
    values_flat.sort()
    coeff = values_flat[-2]
    if coeff > 0.3:
        return sentence_tokens[index]

def main():
    topic = input("Please enter a topic: ")
    try:
        text = wikipedia.page(topic).content
    except wikipedia.exceptions.DisambiguationError as e:
        print("Please choose a more specific topic from the options:")
        for option in e.options:
            print(option)
        return
    except wikipedia.exceptions.PageError:
        print("Topic not found on Wikipedia.")
        return

    while True:
        question = input("Hi, what do you want to know?\n")
        if question.lower() == 'quit':
            break
        output = process(text, question)
        if output:
            print(output)
        else:
            print("I don't know.")

if __name__ == "__main__":
    main()