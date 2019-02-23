import sys, nltk, operator, re
import baseline
import chunk
from nltk.stem.wordnet import WordNetLemmatizer
import constituency
import dependency
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
from rake_nltk import Rake

GRAMMAR =   """
            N: {<PRP>|<NN.*>}
            V: {<V.*>}
            ADJ: {<JJ.*>}
            NP: {<DT>? <ADJ>* <N>+}
            {<DT|PP\$>?<JJ>*<NN>}   
            {<NNP>+}
            PP: {<IN><NP><POS>?<N>*}
            VP: {<TO>? <V> (<NP>)*}
            RP: {<PP><VP>?<ADJ|VP>?<NP>?}
            """

LOC_PP = set(["in", "on", "at"])

PERSONAL_PRONOUN=set(["he","she"])
def dependent(question,story):
    qgraph = question["dep"]
    question_text=question["text"]
    question_text=chunk.get_sentences(question_text)
    question_prefix=question_text[0][0][0]
 
    
    #print("qgraph:", qgraph)

    # The answer is in the second sentence
    # You would have to figure this out like in the chunking demo
    if question["type"]=='Sch':
        sgraph = story["sch_dep"][get_Index(question,story)]
    else:
        sgraph = story["story_dep"][get_Index(question,story)]
    #print(sgraph)
    
    lmtzr = WordNetLemmatizer()
    #for node in sgraph.nodes.values():
    #    tag = node["tag"]
    #    word = node["word"]
    #    if word is not None:
    #        if tag.startswith("V"):
    #            print(lmtzr.lemmatize(word, 'v'))
    #        else:
    #            print(lmtzr.lemmatize(word, 'n'))
    #print()
    
    answer = dependency.find_answer(qgraph, sgraph)
    print("Type:", question["type"])
    print("question:", question["text"])
    print("answer:", answer)
    print()
    return answer
def get_Index(question,story):
    real_question = question
    question_id = question["qid"]

    if question['type']=='Sch':
        text=story['sch']
    else:
        text = story["text"]
    question = question["text"]
    #print("QUESTION: ", question)

    #Code
    stopwords = set(nltk.corpus.stopwords.words("english"))
    #question_stem_list = chunk.lemmatize(nltk.pos_tag(nltk.word_tokenize(question)))
    #question_stem = "".join(t[0] + " " for t in question_stem_list)
    question_stem = question
    qbow = baseline.get_bow(baseline.get_sentences(question_stem)[0], stopwords)
    sentences = baseline.get_sentences(text)
    question=chunk.get_sentences(question)
    base_ans, index = baseline.baseline(qbow, sentences, stopwords)
    return index
def base(question, story):
    #Base
    real_question = question
    question_id = question["qid"]

    if question['type']=='Sch':
        text=story['sch']
    else:
        text = story["text"]
    question = question["text"]
    print("QUESTION: ", question)

    #Code
    stopwords = set(nltk.corpus.stopwords.words("english"))
    #question_stem_list = chunk.lemmatize(nltk.pos_tag(nltk.word_tokenize(question)))
    #question_stem = "".join(t[0] + " " for t in question_stem_list)
    question_stem = question
    qbow = baseline.get_bow(baseline.get_sentences(question_stem)[0], stopwords)
    sentences = baseline.get_sentences(text)
    question=chunk.get_sentences(question)
    base_ans, index = baseline.baseline(qbow, sentences, stopwords)
    newanswer ="".join(t[0]+" " for t in base_ans)
    chunker = nltk.RegexpParser(GRAMMAR)
    tempanswer=chunk.get_sentences(newanswer)
    atree=chunker.parse(tempanswer[0])
    what_set = ["happened", "do"] #this should probably be changed in the future
    what_set = set(what_set)
    rake =Rake()
    rake.extract_keywords_from_text(real_question["text"])
    if question[0][0][0].lower()=="who":

        pos_phrases = nltk.pos_tag(rake.get_ranked_phrases())
        #print(pos_phrases)

        only_noun_pos_phrases = [noun for noun in pos_phrases if re.search(r"NN", noun[1])]
        only_noun_phrases = []
        for i in only_noun_pos_phrases:
            only_noun_phrases.append(i[0])

        np=chunk.find_nounphrase(atree)
        temp_ans=""
        if (np != []):

            counter = 0
            while True:
                temp_ans = ""
                val = False

                for token in np[counter].leaves():

                    temp_ans=temp_ans+" "+token[0]


                for word in only_noun_phrases:

                        if word in temp_ans:
                            val = True
                if val: # if answer contains a word in only_noun_phrases
                    print("bing")
                    if len(np)-1>counter:
                        counter+=1
                    else:
                        temp_ans = newanswer
                        break
                else:
                    break
        else:
            temp_ans = newanswer
        newanswer=temp_ans


    elif question[0][0][0].lower()=="what":
        #TODO will use dependency in the future as what questions are too hard to figure out wihtout knowing which words are dependent on others.
        if any(word in real_question["text"] for word in what_set):
            pp = chunk.find_verbphrase(atree)
        else:
            pp=chunk.find_nounphrase(atree)
        temp_ans=""
        #print([k.leaves() for k in pp])
        if (pp != []):
            if len(pp)> 1: #fix later
                for token in pp[1].leaves():
                    temp_ans = temp_ans + " " + token[0]
            else:
                for token in pp[0].leaves():
                    temp_ans = temp_ans+" "+token[0]
        else:
            temp_ans = newanswer
        newanswer=temp_ans


    elif question[0][0][0].lower()=="where":
        pp=chunk.find_prepphrases(atree)
        print(pp)
        temp_ans=""
        if (pp != []):
            for token in pp[0].leaves():
                temp_ans=temp_ans+" "+token[0]
        else:
            temp_ans = newanswer
        newanswer=temp_ans

    elif question[0][0][0].lower()=="when":
        pp=chunk.find_times(atree)
        temp_ans=""
        if (pp != []):
            for token in pp[0].leaves():
                temp_ans=temp_ans+" "+token[0]
        else:
            temp_ans = newanswer
        newanswer=temp_ans
    elif question[0][0][0].lower() == "why":
        pp=chunk.find_reasons(atree)
        temp_ans=""
        if (pp != []):
            for token in pp[0].leaves():
                temp_ans=temp_ans+" "+token[0]
        else:
            temp_ans = newanswer
        newanswer=temp_ans
 
    if newanswer.replace(" ","") in PERSONAL_PRONOUN and question[0][0][0].lower()=="who":
        i = index 
        if i > 0:
            previous_sentence=sentences[index-i]
            for word,tag in previous_sentence:
                if tag == "NNP":
                    newanswer=word

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
    return dependent(question, story)

   

#############################################################
###     Dont change the code in this section
#############################################################
class QAEngine(QABase):
    @staticmethod
    def answer_question(question, story):
        answer = get_answer(question, story)
        return answer


def run_qa(evala=False):
    QA = QAEngine(evaluate=evala)
    QA.run()
    QA.save_answers()

#############################################################


def main():
    run_qa(evala=False)
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    f = open("score.txt", "w")
    sys.stdout = f
    score_answers()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()
