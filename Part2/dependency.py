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
    
def get_dependents(node, graph, counter=0):

    results = []
    for item in node["deps"]:
        address = node["deps"][item][0]
        dep = graph.nodes[address]
        results.append(dep)
        counter += 1
        if counter >30:
            return results
        else:
            results = results + get_dependents(dep, graph, counter)
    return results


def get_direct_object(qgraph):
    for node in qgraph.nodes.values():
        if node['rel'] == 'nsubj':
            return node
    return None

def find_answer_what_direct(qgraph, sgraph, dataarr, direct_object):
    qmain = find_main(qgraph)
    qword = direct_object["word"]
    posarr = dataarr[0]
    keywords = dataarr[1]
    blacklist = dataarr[2]
    results = []

    snode = find_node(qword, sgraph)

    for node in sgraph.nodes.values():
        if node["word"] is not None:
            if "VB" in nltk.pos_tag([node["word"]])[0][1]:
                for dep in get_dependents(node,sgraph):
                    if dep["word"] == direct_object["word"]:
                        for node2 in sgraph.nodes.values():
                            if node2["rel"] == "dobj":
                                if node2["head"] == node["address"]:
                                    deps3 = get_dependents(node2, sgraph)
                                    deps3 = sorted(deps3 + [node2], key=operator.itemgetter("address"))
                                    return " ".join(dep3["word"] for dep3 in deps3)

        #if nltk.pos_tag(node["word"])[0][1]:
            #for dep in get_dependents(node,sgraph)
        #print("NODE", node)
        #if node.get('head', None) == snode["address"]:






        """
        if node["word"] is not None and node["word"] in direct_object["word"]:
            print(node["word"])
            print(get_dependents(node,sgraph))
            for node2 in sgraph.nodes.values():
                if node2["rel"] == "root":
                    return None

        
            for node2 in sgraph.nodes.values():
                if node2["rel"] == "root":
                    for dep in get_dependents(node2, sgraph):
                        if dep["word"] == node["word"]:
                            for node3 in sgraph.nodes.values():
                                if node3["rel"] == "dobj":
                                    deps = get_dependents(node3, sgraph)
                                    deps = sorted(deps + [node], key=operator.itemgetter("address"))
                                    str =  " ".join(dep["word"] for dep in deps)
                                    str = node2["word"]+ " " + str
                                    return str
            """

def find_answer(qgraph, sgraph, dataarr):
    qmain = find_main(qgraph)
    qword = qmain["word"]
    posarr = dataarr[0]
    keywords = dataarr[1]
    blacklist = dataarr[2]
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
        if keywords == []: return True
        deps = get_dependents(node, sgraph)
        for dep in deps:
            if dep['lemma'] in keywords:
                return True
        return False

    for pos in posarr:
        snode = find_node(qword, sgraph)
        if snode == []:
            return "Snode null"
        for node in sgraph.nodes.values():
            # if node.get('head', None) == snode["address"]:
            if node['rel'] == pos and search_blacklist(node):
                deps = get_dependents(node, sgraph)
                deps = sorted(deps+[node], key=operator.itemgetter("address"))
                return " ".join(dep["word"] for dep in deps)


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

