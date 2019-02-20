import sys, nltk, operator, re
from rake_nltk import Rake
import baseline_stub
from nltk.stem.wordnet import WordNetLemmatizer
import constituency_demo_stub
import dependency_demo_stub
import chunker
from qa_engine.base import QABase
from nltk.stem.wordnet import WordNetLemmatizer
from qa_engine.score_answers import main as score_answers

GRAMMAR =   """
            N: {<PRP>|<NN.*>}
            V: {<V.*>}
            ADJ: {<JJ.*>}
            NP: {<DT>? <ADJ>* <N>+}
            {<DT|PP\$>?<JJ>*<NN>}   
            {<NNP>+}
            PP: {<IN><NP><POS>?<N>*}
            VP: {<TO>? <V> (<NP>|<PP>)*}
            """


LOC_PP = set(["in", "on", "at"])


def baseline(question,story):
     ###     Your Code Goes Here         ###
    question_id = question["qid"]

    driver = QABase()
    q = driver.get_question(question_id)
    story = driver.get_story(q["sid"])
    if question['type']=='sch':
        text=story['sch']
    else:
        text = story["text"]
    question = q["text"]
    print("question:", question)
    

    stopwords = set(nltk.corpus.stopwords.words("english"))

    qbow = baseline_stub.get_bow(baseline_stub.get_sentences(question)[0], stopwords)
    sentences = baseline_stub.get_sentences(text)
    answer = baseline_stub.baseline(qbow, sentences, stopwords)
    newanswer=""
    newanswer =newanswer.join(t[0]+" " for t in answer)
    #print("answer:", " ".join(t[0] for t in answer))

    print()
 
    print(question)
    chunker = nltk.RegexpParser(GRAMMAR)
    question=chunker.get_sentences(question)
    print(question)
    qtree=chunker.parse(question[0])
    #print(question[0][0][0])
    #print(qtree)
    #print()
    tempanswer=newanswer
    tempanswer=chunker.get_sentences(tempanswer)
    atree=chunker.parse(tempanswer[0])
    if question[0][0][0].lower()=="who":
        np=chunker.find_nounphrase(atree)
        print("Noun Phrase")
        #for t in np:
            #print(" ".join([token[0] for token in t.leaves()]))
        answer1=""
        for token in np[0].leaves():
            answer1=answer1+" "+token[0]
        #print(answer1)
        newanswer=answer1
    elif question[0][0][0].lower()=="where":
        pp=chunker.find_locations(atree)
        answer1=""
        print("VERBPHRASE")
        for t in pp:
            print(" ".join([token[0] for token in t.leaves()]))
        for token in pp[0].leaves():
            answer1=answer1+" "+token[0]
        newanswer=answer1

    #print(tempanswer)
    #print(atree)
    print("ANSWER ",newanswer)
    print()


    ###     End of Your Code         ###

    return newanswer
def chunk(q,story):
    chunker = nltk.RegexpParser(GRAMMAR)
    lmtzr = WordNetLemmatizer()
   
    text = story["text"]

    question=q["text"]
    question=chunker.get_sentences(question)
    qtree=chunker.parse(question[0])
  
    np=chunker.find_nounphrase(qtree)
    vp=chunker.find_verbphrase(qtree)
    
    print(vp)
    vp=vp[len(vp)-1]

    print("Noun Phrase")
    for t in np:
        print(" ".join([token[0] for token in t.leaves()]))
    print("Verb Phrase")
    for t in vp:
        print(" ".join([token[0] for token in t.leaves()]))




    # Apply the standard NLP pipeline we've seen before
    sentences = chunker.get_sentences(text)

    # Assume we're given the keywords for now
    # What is happening
    verb = "sitting"
    # Who is doing it
    subj = "crow"
    subj=chunker.get_Subject(np)
    print(chunker.get_Action(vp))
    verb=chunker.get_Action(vp)
    # Where is it happening (what we want to know)
    loc = None
    testsubj=[lmtzr.lemmatize(word,"n")for word in subj]  
    testverb=[lmtzr.lemmatize(word,"v") for word in verb]
    print("TEST VERB IS ",testverb)
    total=testsubj+testverb
    print(total)
    # Might be useful to stem the words in case there isn't an extact
    # string match
    subj_stem = lmtzr.lemmatize(subj[0], "n")
    print("SUBJECT: ",subj_stem)
    verb_stem = lmtzr.lemmatize(verb[0], "v")
    print("VERB:  ",verb_stem)

    # Find the sentences that have all of our keywords in them
    # How could we make this better?
    crow_sentences = chunker.find_sentences(total, sentences)

    # Extract the candidate locations from these sentences
    locations = chunker.find_candidates(crow_sentences, chunker)

    # Print them out
    answer=""
    for loc in locations:
        print("done")
        print(loc)
        print(" ".join([token[0] for token in loc.leaves()]))
        for token in loc.leaves():
            answer=answer+" "+token[0]
    print("ANSWER :"+answer)
    return answer



    

def get_answer(question, story):
    """
    :param question: dict
    :param story: dict
    :return: str


    question is a dictionary with keys:
        dep -- A list of dependency graphs for the question sentence.
        par -- A list of constituency parses for the question sentence.
        text -- The raw text of story.
        sid --  The story id.
        difficulty -- easy, medium, or hard
        type -- whether you need to use the 'sch' or 'story' versions
                of the .
        qid  --  The id of the question.


    story is a dictionary with keys:
        story_dep -- list of dependency graphs for each sentence of
                    the story version.
        sch_dep -- list of dependency graphs for each sentence of
                    the sch version.
        sch_par -- list of constituency parses for each sentence of
                    the sch version.
        story_par -- list of constituency parses for each sentence of
                    the story version.
        sch --  the raw text for the sch version.
        text -- the raw text for the story version.
        sid --  the story id


    """
    ###     Your Code Goes Here         ###
    question_id = question["qid"]
    lmtzr = WordNetLemmatizer()
    driver = QABase()
    q = driver.get_question(question_id)
    story = driver.get_story(q["sid"])
    text = story["text"]

    question = q["text"]
    q_data = answer_type(question)  #An array of question data
    #if q_data[0] in ["Who", "What"]:
    #    return "None"
    print("QUESTION:", question)
    stopwords = set(nltk.corpus.stopwords.words("english"))
    #https://www.quora.com/How-can-I-extract-keywords-from-a-document-using-NLTK
    #if we are allowed to, use rake


    qbow = baseline_stub.get_bow(baseline_stub.get_sentences(question)[0], stopwords)
    sentences = baseline_stub.get_sentences(text)
    print(sentences)


    #https://pypi.org/project/rake-nltk/
    r = Rake()
    r.extract_keywords_from_text(question)
    r.get_ranked_phrases() #right now this can give bigrams and unigrams (possibly more)
    q = question["text"]
    q=nltk.word_tokenize(q)
    print(q)
    #if q[0]=='Where':
    #return chunk(question,story)
    #else:
    return baseline(question,story)



    possible_sentences = chunker.find_sentences(r.get_ranked_phrases(), chunker.get_sentences(text))
    print("POSSIBLE SENTENCES" + str(possible_sentences))
    #perhaps make a way to firtsly rank the possible sentences
    #possibly with pronouns assume that it may be the subject.

    answer = baseline_stub.baseline(qbow, sentences, stopwords)
    #print(answer)

    #Regex based answer
    #answer = " ".join(t[0] for t in answer)
    #answer = shorten_answer(q, answer)   

    #Chunking based answer
    chunker_options = chunker.chunking(question_id, q_data[1], q_data[2])
    
    if chunker_options:
        answer = chunker_options[0]
    else:
        answer = " ".join(t[0] for t in answer)
        answer = shorten_answer(q, answer)   

    print("ANSWER:", answer)
    print()
    
    '''
    #print("Question: " + question["text"])
    reformulate_question(question["text"])
    answer = "placeholder"
    '''
    ###     End of Your Code         ###
    return answer 

def shorten_answer(q, atext): 
    #If a phrase is prefixed with an indicator, immediately return the substring
    qtext = q["text"]
    atype = answer_type(qtext)[0]
    if (atype == "Who"):
        candidates = atext
    if (atype == "What"):
        candidates = atext
    if (atype == "Where"):
        candidates = regex_candidates(["on","in","at"], atext)
    if (atype == "When"):
        candidates = regex_candidates(["on"], atext)
    if (atype == "Why"):
        candidates = regex_candidates(["because", "since", "for", "to"], atext)
    return candidates[0][0: min(len(candidates[0]), 45)]

def regex_candidates(indicators, atext):
    #Attempts to shorten the baseline answer using regex only by detecting indicators
    options = []
    for indicator in indicators:
            reasoning = re.search(r'.* '+indicator+' (.*?) [,.]', atext)
            if reasoning:
                for option in reasoning.groups():
                    options.append(option)
    if len(options):
    #    output = options[0]     #Assuming the prefixes are fed from most important to least, index 0 should be best choice
    #    for option in options:
    #        output = option if len(option) < len(output) else output
    #    return output[0: min(len(output), 45)]
        return options

    #process text differently if it fails prefix detection
    return [atext]

def answer_type(qtext):
    if bool(re.search("[Ww]ho",qtext)): return ("Who", "NP", set())
    if bool(re.search("[Ww]hat",qtext)): return ("What", "NP", set())
    if bool(re.search("[Ww]here",qtext)): return ("Where", "PP", set(["on","in","at"]))
    if bool(re.search("[Ww]hen",qtext)): return ("When", "NP", set(["on"]))
    if bool(re.search("[Ww]hy",qtext)): return ("Why", "PP", set(["because", "since", "for", "to"]))



def reformulate_question(qtext):
    if (answer_type(qtext)[0] == "Who"):
        ref = re.search(r'Who (.*).', qtext).group(1)
        naive_regex = '(.*) '+ ref
        print(naive_regex)

#############################################################
###     Dont change the code in this section
#############################################################
class QAEngine(QABase):
    @staticmethod
    def answer_question(question, story):
        answer = get_answer(question, story)
        return answer


def run_qa():
    QA = QAEngine()
    QA.run()
    QA.save_answers()

#############################################################


def main():
    run_qa()
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    scores = open("scores.txt", "w")
    sys.stdout = scores
    score_answers()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()
