import sys, nltk, operator
import baseline_stub
import chunk_demo
from nltk.stem.wordnet import WordNetLemmatizer
import constituency_demo_stub
import dependency_demo_stub
from qa_engine.base import QABase
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
        print("USING SCHEHZARD")
        text=story['sch']
    else:
        text = story["text"]
    question = q["text"]
    print("QUESTION", question)
    

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
    question=chunk_demo.get_sentences(question)
    tempanswer=newanswer
    tempanswer=chunk_demo.get_sentences(tempanswer)
    atree=chunker.parse(tempanswer[0])
    if question[0][0][0].lower()=="who":
        np=chunk_demo.find_nounphrase(atree)
        answer1=""
        for token in np[0].leaves():
            answer1=answer1+" "+token[0]
        newanswer=answer1

    elif question[0][0][0].lower()=="where":
        pp=chunk_demo.find_locations(atree)
        answer1=""
        for token in pp[0].leaves():
            answer1=answer1+" "+token[0]
        newanswer=answer1

    print("ANSWER ",newanswer)
    print()
    return newanswer

def chunk(q,story):
    chunker = nltk.RegexpParser(GRAMMAR)
    lmtzr = WordNetLemmatizer()
   
    text = story["text"]

    question=q["text"]
    question=chunk_demo.get_sentences(question)
    qtree=chunker.parse(question[0])
  
    np=chunk_demo.find_nounphrase(qtree)
    vp=chunk_demo.find_verbphrase(qtree)
    
    print(vp)
    vp=vp[len(vp)-1]

    print("Noun Phrase")
    for t in np:
        print(" ".join([token[0] for token in t.leaves()]))
    print("Verb Phrase")
    for t in vp:
        print(" ".join([token[0] for token in t.leaves()]))




    # Apply the standard NLP pipeline we've seen before
    sentences = chunk_demo.get_sentences(text)

    # Assume we're given the keywords for now
    # What is happening
    verb = "sitting"
    # Who is doing it
    subj = "crow"
    subj=chunk_demo.get_Subject(np)
    print(chunk_demo.get_Action(vp))
    verb=chunk_demo.get_Action(vp)
    # Where is it happening (what we want to know)
    loc = None
    testsubj=[lmtzr.lemmatize(word,"n")for word in subj]  
    testverb=[lmtzr.lemmatize(word,"v") for word in verb]
    total=testsubj+testverb
    print(total)
    # Might be useful to stem the words in case there isn't an extact
    # string match
    subj_stem = lmtzr.lemmatize(subj[0], "n")
    verb_stem = lmtzr.lemmatize(verb[0], "v")

    # Find the sentences that have all of our keywords in them
    # How could we make this better?
    crow_sentences = chunk_demo.find_sentences(total, sentences)

    # Extract the candidate locations from these sentences
    locations = chunk_demo.find_candidates(crow_sentences, chunker)

    # Print them out
    answer=""
    for loc in locations:
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
    return baseline(question,story)

   



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
    score_answers()

if __name__ == "__main__":
    main()
