import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from colorama import Fore, init
import re

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


# function to print sentiments of the sentence.
def sentiment_scores(sentence):
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)

    print("Overall sentiment dictionary is : ", sentiment_dict)
    print(Fore.RED + "sentence was rated as ", sentiment_dict['neg'] * 100, "% Negative")
    print(Fore.BLUE + "sentence was rated as ", sentiment_dict['neu'] * 100, "% Neutral")
    print(Fore.GREEN + "sentence was rated as ", sentiment_dict['pos'] * 100, "% Positive")

    highlighted_text = highlight_words(sentence)
    print("\n" + highlighted_text)

    print("Sentence Overall Rated As", end=" ")

    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05:
        print(Fore.GREEN + "Positive")
    elif sentiment_dict['compound'] <= - 0.05:
        print(Fore.RED + "Negative")
    else:
        print(Fore.BLUE + "Neutral")


def highlight_words(text):
    sid = SentimentIntensityAnalyzer()
    negative_words = set()
    positive_words = set()

    # Tokenize the text and remove punctuation
    words = word_tokenize(text)
    tokens = [re.sub(r'[^\w\s]', '', token) for token in words]

    # Filter out stopwords
    tokens = [token for token in tokens if token.lower() not in stopwords.words('english')]

    # Identify negatively sounding words using VADER
    for token in tokens:
        sentiment_scores = sid.polarity_scores(token)
        if sentiment_scores['neg'] > sentiment_scores['pos']:
            negative_words.add(token)
        elif sentiment_scores['pos'] > sentiment_scores['neg']:
            positive_words.add(token)

    # Highlight negative words in the original text
    highlighted_text = []
    for token in words:
        if token in negative_words:
            highlighted_text.append(Fore.RED + f"{token}" + Fore.RESET)
        elif token in positive_words:
            highlighted_text.append(Fore.GREEN + f"{token}" + Fore.RESET)
        else:
            highlighted_text.append(token)

    return ' '.join(highlighted_text)


# Driver code
if __name__ == "__main__":

    init(autoreset=True)

    while True:
        sentence = input("\nEnter a sentence (or 'q' to quit): ")
        if sentence.lower() == 'q':
            exit(0)
        else:
            print("You entered:", sentence)

        sentiment_scores(sentence)
