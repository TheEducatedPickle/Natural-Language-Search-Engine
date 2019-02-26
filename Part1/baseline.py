#!/usr/bin/env python
'''
Created on May 14, 2014
@author: reid

Modified on May 21, 2015
'''
from nltk.stem.wordnet import WordNetLemmatizer
import sys, nltk, operator
from qa_engine.base import QABase
<<<<<<< HEAD:baseline_stub.py

#def get_SurroundingSentences(text):

=======
>>>>>>> posmap:Part1/baseline.py
    
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
        
        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        overlap = len(qbow & sbow)
        
        answers.append((overlap, sent, number))
        number += 1
    # Sort the results by the first element of the tuple (i.e., the count)
    # Sort answers from smallest to largest by default, so reverse it
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)

    #for sent in best_candidate_sents: #Filter using Rake
    #    ans_tagged = set([word[0] for word in sent])
    #    best_overlap = len(ans_tagged & qbow)
    #    candidates_ranked_on_rake.append((ans_tagged, best_overlap))
    #print(candidates_ranked_on_rake)
    #best_answer = sorted(candidates_ranked_on_rake, key=takeSecond)

    # Return the best answer
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
    print(find_phrase(get_sentences(question)[0],qbow))
    sentences = get_sentences(text)
    answer = baseline(qbow, sentences, stopwords)
    print("answer:", " ".join(t[0] for t in answer))
