'''
This module attempts to classify the answer type of
a given question.

'''
from collections import Counter
import cPickle as pickle
import itertools
import nlp_parser
import nltk
import random
import re
import sys


def extract_candidate_answers(passage, answer_type, question, stop_words):
    '''
    Attempts to return a list of possible answers from the passage
    
    Answer types we need to support: PERSON, LOCATION, ORGANIZATION, NUMBER, DATE, FACT
    Answer types we currently support: PERSON, LOCATION, ORGANIZATION, NUMBER (very poorly), DATE (slightly less poorly)
    
    
    First attempt: For our first attempt we just use the nltk.ne_chunk to 
    crudely tag named entities. We then return all the entities equal to the answer type
    (Which is why we only support PERSON, LOCATION, ORGANIZATION)
    
    Second attempt: very similar to first attempt, but we include Number and Date. Numbers are found by
    using nltk to tag words since it has a tag for a number. We first however find relevenat snippets of the passage
    first. We do this as:
    foreach word in question that is not a stop_word:
        foreach occurrence of word in passage
            add [5 previous words, word, 5 next words] to snippet.
    '''
    answer_type = answer_type.upper()
    candidate_answers = []
    
    #Get tokens from question and remove those that are in the stop words
    question_tokens = nltk.word_tokenize(question)
    
    important_terms = [token for token in question_tokens if token.lower() not in stop_words]
    snippets = []
    
    for term in important_terms:
        snippet_regex = r'((?:\S+\s*){,5})(%s)((?:\s*\S+\s*){,5})' % term #Ganked this from some fool on Stack Overflow
        
        for snippet_match in re.finditer(snippet_regex, passage, re.IGNORECASE):
            (before, term, after) = snippet_match.group(1, 2, 3)
            snippets.append(before + term + after)
            
    
    ''' Problem: there could be duplicated snippets. For example if the query was 'Who was the oldest president'
    oldest and president would both be important_terms, so a phrase such as 'John Doe was the oldest president when he took office in 1994'
    would trigger snippets 'John Doe was the oldest president when he took office' and 'John Doe was the oldest president when he took office in'
    '''
    passage = " ".join(snippets)
    tokenized_passage = nltk.word_tokenize(passage)
    tagged_passage =nltk.pos_tag(tokenized_passage)
    
    if answer_type == "DATE":
        ''' Dates are pretty hard to extract because they occur in so many different manners. Common date forms:
            blah (1990 - 2000). 
            2010 (Four digit number)
            April 1st (Month followed by a day)
            1st of April
        '''
        
        month_map = {'Jan' : 'January', 'January' : 'January', 'Feb' : 'February', 'Febr' : 'February', 
                     'February' : 'February', 'March' : 'March', 'Marc' : 'March', 'Mar' : 'March',
                     'April' : 'April', 'Apr' : 'April', 'May' : 'May', 'June' : 'June', 'Jun' : 'June',
                     'July' : 'July', 'Jul' : 'July', 'August' : 'August', 'Aug' : 'August',
                      'Sep' : 'September', 'Sept' : 'September', 'September' : 'September',
                      'October' : 'October', 'Oct': 'October', 'November' : 'November', 'Nov' : 'November',
                      'Dec' : 'December', 'December' : 'December'}
        
        #Attempts to match a month or month abbreviation followed by a day number
        #followed by a year
        month_date_year_regex = r'(Jan|January|Feb|Febr|February|March|Marc|Mar|' + \
                           r'Apr|April|May|June|Jun|Jul|July|Aug|August|' + \
                           r'Sept|Sep|September|October|Oct|Nov|November|Dec|December)' + \
                           r'[,.]?\s*(\d{1,2})(st|nd|rd|th)?,?\s*(\d{4})?' \
                           
        year_regex = r'\D(\d{4})\D' #Match non-digit, then 4 digits, then a non-digit
        
        for month_date_year_match in re.finditer(month_date_year_regex, passage, re.IGNORECASE):
            (month, day, _, year) = month_date_year_match.group(1, 2, 3, 4)
            month = month_map[month]
            if day[0] == '0':
                day = day[1:]
            if day == '':
                continue
            
            date_list = [x for x in [month, day, year] if x is not None]
            
            candidate_answers.append(" ".join(date_list))
            
        for year in re.finditer(year_regex, passage):
            candidate_answers.append(year.group(1)) #Year is the group 1
        
    elif answer_type == 'FACT':
        pass
    elif answer_type == "NUMBER":
        grammar = r"""
        NUM : {<CD><NN.?>}
        {<CD>}
        """
        cp = nltk.RegexpParser(grammar)
        parse_tree = cp.parse(tagged_passage)
        for node in parse_tree:
            if hasattr(node, 'node'): #Must be a number
                candidate_answers.append(" ".join(nltk.untag(node[:])))
        
    
    else:
        ne_passage = nltk.ne_chunk(tagged_passage)
        
        # Right now we change an answer type of LOCATION to GPE
        # since nltk is fucking weird
        if answer_type == 'LOCATION' : answer_type = 'GPE'
        
        for node in ne_passage:
            if hasattr(node, 'node'): #Is a named entity
                if node.node == answer_type:
                    candidate_answers.append(" ".join(nltk.untag(node[:])))
    
    return candidate_answers
    
  
def extract_important_words(question):
    '''
    Trys to extract the important terms from the sentence.
    E.g. 'Who was the first person to visit Antarctica?'
    would hopefully return ['first' 'person', 'visit', 'Antarctica']
    '''
    
    tokens = nltk.word_tokenize(question)
    tags = nltk.pos_tag(tokens)
    
    # Second attempt: we attempt to extract noun phrases as one important term
    # as well as extracting verbs and regular nouns and CDs
    
    grammar = r"""
        NP: {<JJ.>+<NN.?>+}
            {<NN.?>+}
    """
    
    cp = nltk.RegexpParser(grammar)
    parse_tree = cp.parse(tags)
    important_terms = []
    stop_words =  ['is', '\'s', 'did', 'was', 'are']
    
    for node in parse_tree:
        if hasattr(node, 'node'):
            #This is a noun phrase
            phrase = " ".join([word for (word, _) in node])
            important_terms.append(phrase)
        else:
            (word, tag) = node
            if word not in stop_words and (tag.startswith('NN') or tag.startswith('VB') \
                                         or tag.startswith('CD')):
                important_terms.append(word)
            
    '''
    # First attempt, just return all the nouns and verbs and CDs (numbers)
    # while ignoring a hand crafted list of stop words
    
    for (word, tag) in tags:
        if word not in stop_words and (tag.startswith('NN') or tag.startswith('VB') \
                                         or tag.startswith('CD')):
            important_terms.append(word)
    '''
    return important_terms

def build_queries(question):
    '''
    Returns a list of queries to search the documents for.
    
    Right now we just extract the important terms from the question and return
    a list of all possible queries.
    
    Ex: "Who was the oldest president?"
    Important terms: oldest, president
    Queries [oldest, president, oldest president]
    '''
    def is_non_empty(iterable):
        for item in iterable:
            if item:
                return True
        return False
    
    important_terms = extract_important_words(question)
    if len(important_terms) <= 1:
        return important_terms
    important_terms_tuples = [(term, '') for term in important_terms]

    cartesian_product = itertools.product(*important_terms_tuples)
    queries = [" ".join(product).strip() for product in cartesian_product if is_non_empty(product)]
    
    return queries
def answer_features_words_and_pos(tokens, vocab, bigrams):
    if tokens[-1] == '?':
        tokens = tokens[:-1]
        
    tags = [tag for (word, tag) in nltk.pos_tag(tokens)]
    answer_bigrams = zip(tags, tags[:])
    features = {}
    for word in vocab:
        features['has_word(%s)' % word] = word in tokens
        
    for bigram in bigrams:
        features['has_bigram%s' % str(bigram)] = bigram in answer_bigrams

    return features

def build_feature_function(vocab):
    def answer_features_words(tokens):
        if tokens[-1] == '?':
            tokens = tokens[:-1]
        features = {}
        for word in vocab:
            features['has_word(%s)' % word] = word in tokens
            
        features['question_length'] = len(tokens)
        return features
    return answer_features_words
    

def train_classifier(classifier_to_use, training_set):
    '''
    classifier_to_use is an nltk classifier class.
    training_set is a list of training examples of the form
    (feature_set, class)
    '''
    pass
    
    
def tag_answer(answer):
    tokens = nltk.word_tokenize(answer)
    tokens = nltk.pos_tag(tokens)
    return nltk.ne_chunk(tokens)


def build_classifier(tagged_pairs_filename, classifier_type = nltk.DecisionTreeClassifier, output_filename = None, k = None):
    '''
    Builds a classifier based on the classifier_type. Performs k-fold cross validation, prints out
    the accuracy of each fold, as well as average accuracy. Returns the classifier trained
    on the entire training set and pickles it to the output_filename
    @output_filename - File to write classifier and vocab to, if None then we don't write the output
    @param k - Number of folds to do: if k is None then we don't do any folds and just build the classifier
    
    '''
    all_question_entity_pairs = nlp_parser.read_manual_tags(tagged_pairs_filename)
    
    all_question_entity_pairs = [(question, entity) for (question, entities) in all_question_entity_pairs for entity in entities]
    
    random.shuffle(all_question_entity_pairs)
    vocab = set()
    
    for (i, (question, entity)) in enumerate(all_question_entity_pairs):
        question_tokens = nltk.word_tokenize(question)
        if question_tokens[-1] == "?" or question_tokens[-1] == ".":
            question_tokens = question_tokens[:-1]
        
        
        for token in question_tokens:
            vocab.add(token)
            
        all_question_entity_pairs[i] = (question_tokens, entity)
    
    feature_function = build_feature_function(vocab)
    all_question_entity_pairs = [(feature_function(tokens), entity) for (tokens, entity) in all_question_entity_pairs]
    
    if k is not None:
        segment_length = len(all_question_entity_pairs) / k
        start_indices = [segment_length * x for x in range(k)]
        end_indices = [segment_length * x for x in range(1, k + 1)]
        end_indices[-1] = len(all_question_entity_pairs) #Manually set this or else we might lose an element or two from rounding
        
        segments = [all_question_entity_pairs[start:end] for (start, end) in zip(start_indices,end_indices)]
        #Do k-fold cross validation
        accuracies = []
        for k_0 in xrange(k):
            test = segments[k_0]
            training = []
            for i in xrange(k):
                if i == k_0: continue
                training+= [x for x in segments[i]]
            
            classifier = classifier_type.train(training)
            accuracy = nltk.classify.accuracy(classifier, test)
            print 'Accuracy for fold %i : %f' % (k_0, accuracy)
            accuracies.append(accuracy)
            
        print 'Average accuracy : %f' % (sum(accuracies) / k)
        
    #Now build entire classifier
    classifier = classifier_type.train(all_question_entity_pairs)
    
    if output_filename is not None:
        f = open(output_filename, 'wb')
        pickle.dump(classifier, f)
        pickle.dump(vocab, open('answer_type_vocab.p', 'wb'))

    return classifier
    
    
def quiz(classifier_filename, vocab_filename):
    classifier = pickle.load(open(classifier_filename, 'rb'))
    vocab = pickle.load(open(vocab_filename, 'rb'))
    feature_function = build_feature_function(vocab)
    print 'Enter a question to be typed, or enter a blank line to quit'
    
    while True:
        question = sys.stdin.readline().strip()
        if not question: break
        tokens = nltk.word_tokenize(question)
        
        features = feature_function(tokens)
        
        entity = classifier.classify(features).upper()
        
        print entity
def classify_question(classifier, feature_function, question):
    tokens = nltk.word_tokenize(question)
    features = feature_function(tokens)
    entity = classifier.classify(features).upper()
    return entity

def test():
    questions = nlp_parser.read_manual_tags("manually_tagged_questions.txt")
    
    random.shuffle(questions)
    for (question, _ ) in questions[:10]:
        print question
        print nltk.pos_tag(nltk.word_tokenize(question))
        print extract_important_words(question)
        print ''
        
if __name__ == "__main__":
    if sys.argv > 1:
        if sys.argv[1] == "-build":
            if sys.argv > 2:
                k = sys.argv[2]
            else: 
                k = None
            
            build_classifier("manually_tagged_questions.txt", "answer_type_classifier.p", nltk.DecisionTreeClassifier, k)   
        elif sys.argv[1] == '-quiz':
            quiz("answer_type_classifier.p", "answer_type_vocab.p") 
        elif sys.argv[1] == '-queries':
            classifier = pickle.load(open("answer_type_classifier.p", 'rb'))
            vocab = pickle.load(open("answer_type_vocab.p", 'rb'))
            feature_function = build_feature_function(vocab)
            questions = nlp_parser.parse_questions("questions.txt")
            
            random.shuffle(questions)
            
            print "Press enter to type and generate queries for the next question. Enter anything other than a blank line to quit"
            
            for (question_id, question) in questions:
                line = sys.stdin.readline().strip()
                if line != "":
                    break
                
                print "Question : %s" % question
                print "Type : %s " % classify_question(classifier, feature_function, question)
                print "Important terms: %s" % ", ".join(extract_important_words(question))
                print "Queries: %s " % "\n\t".join(build_queries(question))
                print ""
                
        elif sys.argv[1] == '-baseline':
            classifier = pickle.load(open("answer_type_classifier.p", 'rb'))
            stop_words = pickle.load(open('stop_words.p', 'rb'))
            vocab = pickle.load(open("answer_type_vocab.p", 'rb'))
            feature_function = build_feature_function(vocab)
                 
            questions = nlp_parser.parse_questions("questions.txt")
            random.shuffle(questions)
            
            
            types = ['LOCATION', 'NUMBER', 'ORGANIZATION', 'PERSON', 'DATE']
            
            print 'Classifying types : ', types
            for (question_id, question) in questions:
                #First find the question type: 
                type = classify_question(classifier, feature_function, question)
                
                if type not in types:
                    continue
                important_terms = (term for term in nltk.word_tokenize(question) if term.lower() not in stop_words)
                
                
                print 'Question id : ', question_id
                print 'Question : %s' % question
                print 'Non stop words : %s' % " ".join(important_terms)
                print 'Question type : %s' % type
                
                print '\tParsing documents...',
                documents = nlp_parser.parse_documents(question_id)
                print 'Parsed'
                possible_answers = []
                
                print '\tExtracting Answers...',
                for document in documents:
                    possible_answers+= [answer.lower() for answer in extract_candidate_answers(document, type, question, stop_words)]
                print 'Extracted'
                
                counter = Counter(possible_answers)
                print 'Top 5 answers : '
                print counter.most_common(5)
                
                print ""
            
            
            
           
            
        