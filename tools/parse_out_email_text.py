"""
given an opened email file f, parse out all text below the metadata block at the top;
add stemming capabilities) and return a string that contains all the words
in the email (space-separated)
"""
import nltk
from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):

    stemmer = SnowballStemmer("english")
    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

        ### project part 2: comment out the line below
        #words = text_string

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)

        tokens = nltk.word_tokenize(text_string)
        # for i in text_string -> this will break string into letters
        for i in tokens:
            words += stemmer.stem(i) + ' '

    return words

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print(text)

if __name__ == '__main__':
    main()

