#!/usr/bin/env python
'''
Created on May 14, 2014
@author: reid

Modified on May 21, 2015
'''

import sys, nltk, operator
from qa_engine.base import QABase
from rake_nltk import Rake
    
# The standard NLTK pipeline for POS tagging a document
def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    
    return sentences	

def get_bow(tagged_tokens, stopwords):
    return set([t[0].lower() for t in tagged_tokens if t[0].lower() not in stopwords])

def find_phrase(tagged_tokens, qbow):
    for i in range(len(tagged_tokens) - 1, 0, -1):
        word = (tagged_tokens[i])[0]
        if word in qbow:
            return tagged_tokens[i+1:]

def get_ranked_sentences(text):
    r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases() # To get keyword phrases ranked highest to lowest.

# qtokens: is a list of pos tagged question tokens with SW removed
# sentences: is a list of pos tagged story sentences
# stopwords is a set of stopwords
def baseline(qbow, sentences, stopwords):
    # Collect all the candidate answers
    answers = []
    number = 0
    for sent in sentences:
        # A list of all the word tokens in the sentence
        sbow = get_bow(sent, stopwords)
        stext = ""
        for word in sent:
            stext += word[0] + ' '
        most_inf_words = set(get_ranked_sentences(stext))
        
        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        overlap = len(qbow & sbow)
        answers.append((overlap, sent, number))
        number += 1
    # Sort the results by the first element of the tuple (i.e., the count)
    # Sort answers from smallest to largest by default, so reverse it
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)
    # Return the best answer
    most_overlap = answers[0][0]
    best_candidate_sents = []
    candidates_ranked_on_rake = []
    for i in range(0, len(answers)): #Get all sentences that share same overlap
        if answers[i][0] == most_overlap:
            best_candidate_sents.append(answers[i][1])
            #print(" ".join(t[0] for t in answers[i][1]))
        else: break
    for sent in best_candidate_sents:
        best_overlap = len(set([word[0] for word in sent]) & qbow)
        print(best_overlap)

    best_answer = (answers[0])[1]    
    index = (answers[0])[2]
    return best_answer, index


if __name__ == '__main__':

    question_id = "fables-01-1"

    driver = QABase()
    q = driver.get_question(question_id)
    story = driver.get_story(q["sid"])
    text = story["text"]
    question = q["text"]
    print("question:", question)
    stopwords = set(nltk.corpus.stopwords.words("english"))

    qbow = get_bow(get_sentences(question)[0], stopwords)
    sentences = get_sentences(text)
    answer = baseline(qbow, sentences, stopwords)
    print("answer:", " ".join(t[0] for t in answer[0]))
