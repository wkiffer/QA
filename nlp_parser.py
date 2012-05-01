'''
Parses the question and answer data
'''
from collections import defaultdict
import heapq
import nltk
import random
import re
import string
import sys
import cPickle as pickle


def write_manual_tags(tagged_pairs, filename):
    ''' Writes the tagged pairs to filename.
    tagged_pairs is a (question, [Entity_list]) tuple
    and we write to file in the form
    question
    Entity
    Entity
    </que>
    question
    ...
    '''
    f = open(filename, 'w')
    
    for (question, entities) in tagged_pairs:
        f.write(question + "\n")
        for entity in entities:
            f.write(entity + "\n")
            
        f.write("</que>\n")

def parse_documents(question_id):
    '''
    Returns a list of the text from the documents
    relevant to question_id
    '''
    documents = []
    
    f = open('docs/top_docs.' + str(question_id), 'rb')
    doc = ""
    in_doc = False
    for line in f:
        line = line.strip()
        if "<TEXT>" in line:
            line = string.replace(line, "<TEXT>", "")
            doc+= line + " "
            in_doc = True
        elif "</TEXT>" in line:
            line = string.replace(line, "</TEXT>", "")
            doc+= line
            documents.append(doc.strip())
            doc = ""
            in_doc = False
        elif in_doc:
            doc+= line + " "
            
            
    
    return documents
            
def read_manual_tags(filename):
    
    f = open(filename, 'r')
    
    tagged_pairs = []
    tagged_pair_lines = []
    
    for line in f:
        line = line.strip()
        if line == "</que>":
            tagged_pairs.append((tagged_pair_lines[0], tagged_pair_lines[1:]))
            tagged_pair_lines = []
        else:
            tagged_pair_lines.append(line)
    
    return tagged_pairs

def check_validity(tagged_entries):
    '''
    Goes through the tagged entries and maps 
    entity -> How many questions are tagged with that entity
    Then prints out all the entities with less than 5
    tags. Chances are, these are mistakes in entry
    '''
    
    freq_map = {}
    for ( _ , entities) in tagged_entries:
        for entity in entities:
            if entity in freq_map:
                freq_map[entity]+= 1
            else:
                freq_map[entity] = 1
    
    for (entity, freq) in freq_map.iteritems():
        if freq < 5:
            print entity, " : "
        
def manually_tag(all_questions_file, manually_tagged_filename):
    all_questions = parse_questions(all_questions_file)
    already_tagged = read_manual_tags(manually_tagged_filename)
    
    already_tagged_questions = [question for (question, _) in already_tagged]
    
    all_questions = [question for (_, question) in all_questions]
    
    not_tagged = [question for question in all_questions if question not in already_tagged_questions]
    
    print 'You will be presented with a series of questions. For each question, enter the entity' + \
     'that the answer should be. Enter quit if you want to quit and skip to skip a question'
     
    count = 0
    for question in not_tagged:
        print question
        user_input = sys.stdin.readline().strip()
        if len(user_input) == 1:
            user_input = user_input.lower()
            if user_input == 'o':
                user_input = "ORGANIZATION"
            elif user_input == 'f':
                user_input = "FACT"
            elif user_input == 'l':
                user_input = "LOCATION"
            elif user_input == "n":
                user_input = "NUMBER"
            elif user_input == "p":
                user_input = "PERSON"
            elif user_input == 'd':
                user_input = "DATE"
        if user_input == 'quit':
            break
        elif user_input == 'skip':
            continue
        
        count += 1
        entities = [entity.strip().upper() for entity in user_input.split(',')]
        already_tagged.append((question, entities))
    
    print "\n\nTagged %i questions" % count
    print "Total questions tagged: %i" % len(already_tagged)
    
    check_validity(already_tagged)
    
    write_manual_tags(already_tagged, manually_tagged_filename)
        
        
    
def parse_answers(file_name):
    '''
    Parses the answer file. Returns a list of answers. Each answer is of the form
    (question_number, question, [(doc_id, [answers_from_that_doc]), (doc_id, [answers_from_that_doc])...])
    '''
    def is_doc_id(string):
        return re.search('[A-Z]{2}\d+-?\d+', string) is not None
    
    def parse_answer_lines(lines):
        question_number = filter(lambda x: x.isdigit(), answer_lines[0])
        question_text = lines[1].strip()
        lines = lines[2:]
        answer_doc_pairs = []
        doc_id_indices = [x for (x, y) in enumerate(lines) if is_doc_id(y)]
        for (index, doc_id_index) in enumerate(doc_id_indices):
            doc_id = lines[doc_id_index]
            if index + 1 == len(doc_id_indices):
                answers = lines[doc_id_index + 1:]
            else:
                answers = lines[doc_id_index + 1: doc_id_indices[index + 1]]
            
            answer_doc_pairs.append((doc_id, answers))
            
        return (question_number, question_text, answer_doc_pairs)
            
                                                  
    answers = []
    answer_lines = []
    
    f = open(file_name, 'r')
    
    for line in f.readlines():
        line = line.strip()
        if not line and answer_lines:
            answers.append(parse_answer_lines(answer_lines))
            answer_lines = []
        else:
            answer_lines.append(line)
            
    return answers

def generate_stop_words():
    questions = parse_questions('trec_9_questions.txt')
    
    freq_map = defaultdict(lambda : 0)
    for ( _, question) in questions:
        tokens = nltk.word_tokenize(question)
        for token in tokens:
            freq_map[token] += 1
        
    heap = []
    
    for (token, freq) in freq_map.iteritems():
        heapq.heappush(heap, (-1 * freq, token))
    
    stop_words = []
    for _ in xrange(25):
        (freq, stop_word) = heapq.heappop(heap)
        print '%s \t %i' % (stop_word.lower() , freq * -1)
        stop_words.append(stop_word.lower())
    
    pickle.dump(stop_words, open('stop_words.p', 'wb'))
        
    
def parse_questions(file_name):
    '''
    Parses the question file. Returns a list of questions. Each question is of the form
    (number, text)
    '''
    def parse_question_lines(question_lines):
        '''
        Parses a question lines of the form ['<num> Number: 397', 'This is the question']
        '''
        question_number = filter(lambda x: x.isdigit(), question_lines[0])
        question_text = question_lines[1].strip()
        
        return (question_number, question_text)
    
    f = open(file_name, 'r')
    
    questions = []
    question_lines = []
    
    
    for line in f.readlines():
        line = line.strip()
        if not line or line == '<top>' or line == '<desc> Description:':
            continue
        elif line == "</top>":
            questions.append(parse_question_lines(question_lines))
            question_lines = []
        else:
            question_lines.append(line)  
             
    return questions
    


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "-check":
            tagged_entries = read_manual_tags("manually_tagged_questions.txt")
            check_validity(tagged_entries)
            return
        elif sys.argv[1] == '-stop_words':
            generate_stop_words()
            return
    
    else:
        manually_tag("trec_9_questions.txt", "manually_tagged_questions.txt")

if __name__ == "__main__":
    main()
#questions = parse_questions("questions.txt")

