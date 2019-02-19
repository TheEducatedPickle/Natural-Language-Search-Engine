import sys, nltk, operator, re
from rake_nltk import Rake
import baseline_stub
import constituency_demo_stub
import dependency_demo_stub
import chunker
from qa_engine.base import QABase
from nltk.stem.wordnet import WordNetLemmatizer
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

    possible_sentences = chunk_demo.find_sentences(r.get_ranked_phrases(), chunk_demo.get_sentences(text))
    print("POSSIBLE SENTENCES" + str(possible_sentences))
    #perhaps make a way to firtsly rank the possible sentences
    #possibly with pronouns assume that it may be the subject.

    answer = baseline_stub.baseline(qbow, sentences, stopwords)
    #print(answer)

    #Regex based answer
    #answer = " ".join(t[0] for t in answer)
    #answer = shorten_answer(q, answer)   

    #Chunking based answer
    chunker_options = chunker.chunking(question_id,q_data[1], q_data[2])
    
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
