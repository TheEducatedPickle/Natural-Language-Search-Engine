#!/usr/bin/env python
'''
Created on May 14, 2014
@author: reid

Modified on May 21, 2015
'''
import time
import sys, nltk, operator
from qa_engine.base import QABase
import chunk
import qa
import spacy,re
from nltk.stem.wordnet import WordNetLemmatizer
LMTZR = WordNetLemmatizer()
nlp = spacy.load('en_core_web_lg')
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


PERSONAL_PRONOUNS=set(["he","she","it"])
GROUP_PRONOUNS=set(["they"])
def get_candidate(min, sent_index, sentences, tags):
    candidates = []
    for i in reversed(range(min, sent_index)):
        sent = sentences[i]
        for word, tag in sent:
            if tag in tags:
                candidates.append(word)
    return candidates[0] if candidates != [] else None

def sub_proper_nouns(sentences, n=2):
    for i in range(0, len(sentences)):
        sent = sentences[i]
        minimum = max(i-n,0)
        for j in range (0, len(sent)):
            word = sent[j][0]
            tag = sent[j][1]
            if word in PERSONAL_PRONOUNS:
                candidate = get_candidate(minimum, i, sentences, ["NN","NNP"])
                sentences[i][j] = (candidate if candidate != None else word, tag)
            if word in GROUP_PRONOUNS:
                candidate = get_candidate(minimum, i, sentences, ["NNS","NNPS"])
                sentences[i][j] = (candidate if candidate != None else word, tag)
    return sentences

def baseline(qbow, sentences, stopwords,question):
    # Collect all the candidate answers

    sentences = sub_proper_nouns(sentences)
    answers = []
    number = 0
    for sent in sentences:
        # A list of all the word tokens in the sentence
        sbow = get_bow(sent, stopwords)
        #print(qbow)
    
        pos_sbow=nltk.pos_tag(sbow)
        

        temp=[]
        for word,tag in pos_sbow:
            ###
            if re.search("VB", tag):
                temp.append((LMTZR.lemmatize(word, "v")))
            elif re.search("NN",tag):
                temp.append((LMTZR.lemmatize(word,"n")))
            else:
                temp.append(word)
        
        sbow=set(temp)

        pos_qbow=nltk.pos_tag(qbow)
        temp=[]
        for word,tag in pos_qbow:
            if re.search("VB", tag):
                temp.append((LMTZR.lemmatize(word, "v")))
            elif re.search("NN",tag):
                temp.append((LMTZR.lemmatize(word,"n")))
            else:
                temp.append(word)
        qbow=set(temp)

        #print(sbow)
        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        overlap = len(qbow & sbow)
        testoverlap=0
        overlapp= list(qbow & sbow)
        overlapp= nltk.pos_tag(overlapp)
        for word,tag in overlapp:
            if re.search("V",tag):
                testoverlap+=2
            else:
                testoverlap+=1
        
        answers.append((testoverlap, sent, number))
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
    top_answers=[]
    max_overlap=(answers[0])[0]
    for answer in answers:
        if answer[0]==max_overlap:
            top_answers.append(answer)
    similarity=0
    best_answer=""
    question=nltk.word_tokenize(question)
    question[0]=""
    length=len(question)-1
    question[length]=""
    stopwords = set(nltk.corpus.stopwords.words("english"))
    stopwords.add('was')
    question = " ".join(word for word in question if word not in stopwords)
    question=nlp(question)
    #print(top_answers)
    if(len(top_answers)>1):
        #print("in here")
        #print(top_answers[0])
        for answer in top_answers:
           
            sentence= " ".join(t[0] for t in answer[1])
            temp_sim=question.similarity(nlp(sentence))
            if temp_sim>similarity:
                best_answer=answer[1]
                index=answer[2]
                similarity=temp_sim
            
            #print("sentence:", " ".join(t[0] for t in answer)," ",temp_sim)
        return best_answer,index


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
    #print(find_phrase(get_sentences(question)[0],qbow))
    sentences = get_sentences(text)
    answer = baseline(qbow, sentences, stopwords)
    print("answer:", " ".join(t[0] for t in answer))
