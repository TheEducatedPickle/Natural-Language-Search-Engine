import re, sys, nltk, operator
import baseline_stub
import constituency_demo_stub
import dependency_demo_stub
import chunker
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
    question_id = question["qid"]

    driver = QABase()
    q = driver.get_question(question_id)
    story = driver.get_story(q["sid"])
    text = story["text"]

    question = q["text"]
    if answer_type(question) in ["Who", "What"]:
        return "None"
    print("QUESTION:", question)
    stopwords = set(nltk.corpus.stopwords.words("english"))

    qbow = baseline_stub.get_bow(baseline_stub.get_sentences(question)[0], stopwords)
    sentences = baseline_stub.get_sentences(text)
    answer = baseline_stub.baseline(qbow, sentences, stopwords)
    #print(answer)
    answer = " ".join(t[0] for t in answer)
    print("ANSWER:", prefix_shorten_answer(question, answer))
    print()
    
    '''
    #print("Question: " + question["text"])
    reformulate_question(question["text"])
    answer = "placeholder"
    '''
    ###     End of Your Code         ###
    return answer       

def prefix_shorten_answer(qtext, atext): #If a phrase is prefixed with an indicator, immediately return the substring
    atext.replace(' .','')  #TODO fix
    atype = answer_type(qtext)
    if (atype == "Where"):
        return prefix_reduce_text(["on","in","at"], atext)
    if (atype == "When"):
        return prefix_reduce_text(["on"], atext)
    if (atype == "Why"):
        return prefix_reduce_text(["because", "since", "for", "to"], atext)

def prefix_reduce_text(indicators, atext):
    options = []
    for indicator in indicators:
            reasoning = re.search(r'.* '+indicator+' (.*?) [,.]', atext)
            if reasoning:
                for option in reasoning.groups():
                    options.append(option)
    if len(options):
        output = options[0]     #Assuming the prefixes are fed from most important to least, index 0 should be best choice
        #for option in options:
        #    output = option if len(option) < len(output) else output
        return output[0: min(len(output), 45)]

    #process text differently if it fails prefix detection
    return atext

def answer_type(qtext):
    if bool(re.search("[Ww]ho",qtext)): return "Who"
    if bool(re.search("[Ww]hat",qtext)): return "What"
    if bool(re.search("[Ww]here",qtext)): return "Where"
    if bool(re.search("[Ww]hen",qtext)): return "When"
    if bool(re.search("[Ww]hy",qtext)): return "Why"    



def reformulate_question(qtext):
    if (answer_type(qtext) == "Who"):
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
