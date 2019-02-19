'''
Created on May 14, 2014
@author: Reid Swanson

Modified on May 21, 2015
'''

import re, sys, nltk
from nltk.stem.wordnet import WordNetLemmatizer
from qa_engine.base import QABase




# Our simple grammar from class (and the book)
GRAMMAR =   """
            N: {<PRP>|<NN.*>}
            V: {<V.*>}
            ADJ: {<JJ.*>}
            NP: {<DT>? <ADJ>* <N>+}
            PP: {<IN> <NP>}
            VP: {<TO>? <V> (<NP>|<PP>)*}
            """

INDICATORS = set(["a"])

def chunking(question_id, pos, filters):
    # Our tools
    global INDICATORS
    INDICATORS = filters
    chunker = nltk.RegexpParser(GRAMMAR)
    lmtzr = WordNetLemmatizer()

    driver = QABase()
    q = driver.get_question(question_id)
    story = driver.get_story(q["sid"])
    text = story["text"]
    #print("Question " + str(q["text"]))
    sentences = get_sentences(text)

    question_tagged = get_sentences(q["text"])[0]
    subj = get_noun(question_tagged)
    verb = get_verb(question_tagged)

    loc = None
    
    # Might be useful to stem the words in case there isn't an extact
    # string match
    subj_stem = lmtzr.lemmatize(subj, "n")
    verb_stem = lmtzr.lemmatize(verb, "v")
    
    # Find the sentences that have all of our keywords in them
    # How could we make this better?
    subj_sentences = find_sentences([subj_stem, verb_stem], sentences)
    
    # Extract the candidate locations from these sentences
    candidates = find_candidates(subj_sentences, chunker, pos)
    
    # Print them out
    output = []
    for loc in candidates:
        output.append(" ".join([token[0] for token in loc.leaves()]))
    #print(output)
    return output

def get_noun(tagged_sent):
    for pair in tagged_sent:
        if bool(re.search(r'NN', pair[1])):
            return pair[0]
    return ''

def get_verb(tagged_sent):
    for pair in tagged_sent:
        if bool(re.search(r'VB', pair[1])):
            return pair[0]
    return ''

def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

def search_tree(tree, phrase):
    # Starting at the root of the tree
    # Traverse each node and get the subtree underneath it
    # Filter out any subtrees who's label is not a PP
    # Then check to see if the first child (it must be a preposition) is in
    # our set of locative markers
    # If it is then add it to our list of candidate locations
    
    # How do we modify this to return only the NP: add [1] to subtree!
    # How can we make this function more robust?
    # Make sure the crow/subj is to the left
    locations = []
    for subtree in tree.subtrees(filter=(lambda subtree: subtree.label() == phrase)):
        if INDICATORS:
            if (subtree[0][0] in INDICATORS):
                locations.append(subtree)
        else:
            locations.append(subtree)
    
    return locations

def find_candidates(sentences, chunker, phrase):
    candidates = []
    for sent in sentences:
        tree = chunker.parse(sent)
        locations = search_tree(tree, phrase)
        candidates.extend(locations)
        
    return candidates

def find_sentences(patterns, sentences):
    # Get the raw text of each sentence to make it easier to search using regexes
    raw_sentences = [" ".join([token[0] for token in sent]) for sent in sentences]
    
    result = []
    for sent, raw_sent in zip(sentences, raw_sentences):
        for pattern in patterns:
            if not re.search(pattern, raw_sent):
                matches = False
            else:
                matches = True
        if matches:
            result.append(sent)
            
    return result

if __name__ == '__main__':
    # Our tools
    print(chunking("fables-01-1", "PP", set(["on"])))