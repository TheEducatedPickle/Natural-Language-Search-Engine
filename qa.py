import re, sys, nltk, operator
import baseline_stub
import chunk_demo
import constituency_demo_stub
import dependency_demo_stub
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers


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
    '''
    question_id = question["qid"]

    driver = QABase()
    q = driver.get_question(question_id)
    story = driver.get_story(q["sid"])
    text = story["text"]
    question = q["text"]
    print("question:", question)
    stopwords = set(nltk.corpus.stopwords.words("english"))

    qbow = baseline_stub.get_bow(baseline_stub.get_sentences(question)[0], stopwords)
    sentences = baseline_stub.get_sentences(text)
    answer = baseline_stub.baseline(qbow, sentences, stopwords)
    print("answer:", " ".join(t[0] for t in answer))
    print()
    '''
    #print("Question: " + question["text"])
    answer_type(question["text"])
    answer = "placeholder"
    ###     End of Your Code         ###
    return answer

def answer_type(qtext):
    outTypes = {}
    outTypes["Person"] = "Noun"
    outTypes["Object"] = "Noun"
    outTypes["Location"] = "Noun"
    outTypes["Color"] = "Noun"
    outTypes["Time"] = "Time"
    outTypes["Method"] = "Method"

    if bool(re.match("Who",qtext)): return "Who"
    if bool(re.match("What",qtext)): return "What"
    if bool(re.match("Where",qtext)): return "Where"
    if bool(re.match("When",qtext)): return "When"
    if bool(re.match("Why",qtext)): return "Why"    
       

def reformulate_question(qtext):
    if (answer_type(qtext) == "Who"):
        ref = re.search("Who (.*)", qtext)
        regex = "(.*) "+ref

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
    #score_answers()

if __name__ == "__main__":
    main()
