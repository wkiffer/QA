'''
Parses the question and answer data
'''
import re


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
    

#questions = parse_questions("questions.txt")

