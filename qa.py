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
