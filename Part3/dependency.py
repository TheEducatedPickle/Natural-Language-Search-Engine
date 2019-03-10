#!/usr/bin/env python

import re, sys, nltk, operator
from nltk.stem.wordnet import WordNetLemmatizer

from qa_engine.base import QABase
    
def find_main(graph):
    for node in graph.nodes.values():
        if node['rel'] == 'root':
            return node
    return None
    
def find_node(word, graph):
    for node in graph.nodes.values():
        if node["word"] == word:
            return node
    return None
    
def get_dependents(node, graph):
    results = []
    for item in node["deps"]:
        address = node["deps"][item][0]
        dep = graph.nodes[address]
        if dep['rel'] == node['rel']: 
            break
        results.append(dep)
        results = results + get_dependents(dep, graph)
    return results

def find_answer(qgraph, sgraph, dataarr):
    #print(sgraph) 
    qmain = find_main(qgraph)
    qword = qmain["word"]
    posarr = dataarr[0]
    keywords = dataarr[1]
    blacklist = dataarr[2]
    keydeplist = dataarr[3]
    results = []
    def search_blacklist(node): #Searches a node & dependencies for keywords / blacklist
        if node['lemma'] in blacklist:
            return False
        deps = get_dependents(node, sgraph)
        for dep in deps:
            if dep['lemma'] in blacklist:
                return False
        return True
    
    def search_keywords(node): #Searches a node & dependencies for keywords / blacklist
        if keywords == []:
            return True
        #if node['tag'] in ['NNP','NNPS']: return True
        if node['lemma'] in keywords:
            return True
        deps = get_dependents(node, sgraph)
        #if (len(deps) == 0 and node['word'][0] >= 'A' and node['word'][0] <= 'Z'): #Return proper nouns
        #    return True
        for dep in deps:
            if dep['lemma'] in keywords or dep['rel'] in keydeplist:
                return True
        return False

    for pos in posarr:
        snode = find_node(qword, sgraph)
        if snode == []:
            return "Snode null"
        for node in sgraph.nodes.values():
            # if node.get('head', None) == snode["address"]:
            if node['rel'] == pos and search_blacklist(node) and search_keywords(node):
                deps = get_dependents(node, sgraph)
                deps = sorted(deps+[node], key=operator.itemgetter("address"))
                joined = " ".join(dep["word"] for dep in deps)
                joined = re.sub(r'[!.?]','',joined)
                return joined
                
if __name__ == '__main__':
    driver = QABase()

    # Get the first question and its story
    q = driver.get_question("fables-01-1")
    story = driver.get_story(q["sid"])

    # get the dependency graph of the first question
    qgraph = q["dep"]
    print(q["text"])
    #print("qgraph:", qgraph)

    # The answer is in the second sentence
    # You would have to figure this out like in the chunking demo
    print(len(story["sch_dep"]))
    sgraph = story["sch_dep"][1]

    
    lmtzr = WordNetLemmatizer()
    for node in sgraph.nodes.values():
        tag = node["tag"]
        word = node["word"]
        if word is not None:
            if tag.startswith("V"):
                print(lmtzr.lemmatize(word, 'v'))
            else:
                print(lmtzr.lemmatize(word, 'n'))
    print()

    answer = find_answer(qgraph, sgraph, "nsubj")
    print("answer:", answer)

