import sent2vec
import sys
""" from nltk.tokenize import sent_tokenize
from nltk.tokenize.stanford import StanfordTokenizer
import re
import os """
from scipy import spatial
import numpy
import math

""" def tokenize(tknzr, sentence, to_lower=True):
    Arguments:
        - tknzr: a tokenizer implementing the NLTK tokenizer interface
        - sentence: a string to be tokenized
        - to_lower: lowercasing or not

    sentence = sentence.strip()
    sentence = ' '.join([format_token(x) for x in tknzr.tokenize(sentence)])
    if to_lower:
        sentence = sentence.lower()
    sentence = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>',sentence) #replace urls by <url>
    sentence = re.sub('(\@ [^\s]+)','<user>',sentence) #replace @user268 by <user>
    filter(lambda word: ' ' not in word, sentence)
    return sentence

def format_token(token):
    """"""
    if token == '-LRB-':
        token = '('
    elif token == '-RRB-':
        token = ')'
    elif token == '-RSB-':
        token = ']'
    elif token == '-LSB-':
        token = '['
    elif token == '-LCB-':
        token = '{'
    elif token == '-RCB-':
        token = '}'
    return token

def tokenize_sentences(tknzr, sentences, to_lower=True):
    Arguments:
        - tknzr: a tokenizer implementing the NLTK tokenizer interface
        - sentences: a list of sentences
        - to_lower: lowercasing or not
    
    return [tokenize(tknzr, s, to_lower) for s in sentences] """


def QuestionGenCos(A,Q):
   cosineSimilarity = float("-inf")
   AIndex = 0
   #magnitudeQ = numpy.linalg.norm(numpy.array(Q))
   for (i,sentences) in enumerate(A):
      """ sentence = numpy.array(sentences)
      dotProd = numpy.dot(sentence, numpy.array(Q))
      magnitudeA = numpy.linalg.norm(sentence)
      prodMagAQ = float(magnitudeQ) * float(magnitudeA)
      cosSim = float(dotProd)/float(prodMagAQ) """
      cosSim = 1 - spatial.distance.cosine(sentences, Q)
      if cosSim > cosineSimilarity:
         cosineSimilarity = cosSim
         AIndex = i
   return AIndex


		

fileName = sys.argv[1]
questionFile = sys.argv[2]

SNLP_TAGGER_JAR = "stanford-postagger.jar"

sentences = []
with open(fileName, 'r') as fileinput:
   for line in fileinput:
       """ paraSentences = sent_tokenize(line)
       for sentence in paraSentences: """
       sentences.append(line)

""" tknzr = StanfordTokenizer(SNLP_TAGGER_JAR, encoding='utf-8')
s = ' <delimiter> '.join(sentences)
tokenized_sentences_SNLP = tokenize_sentences(tknzr, [s])
tokenized_sentences_SNLP = tokenized_sentences_SNLP[0].split(' <delimiter> ') """

questions = []
with open(questionFile, 'r') as fileinput:
   for line in fileinput:
      questions.append(line)

""" t = ' <delimiter> '.join(questions)
tokenized_sentences_SNLP2 = tokenize_sentences(tknzr, [t])
tokenized_sentences_SNLP2 = tokenized_sentences_SNLP2[0].split(' <delimiter> ') """
model = sent2vec.Sent2vecModel()
model.load_model('wiki_bigrams.bin') # The model can be sent2vec or cbow-c+w-ngrams
articlevec = model.embed_sentences(sentences) 
answers = []
for question in questions:
    questionvec = model.embed_sentence(question)[0]
    index = QuestionGenCos(articlevec, questionvec)
    answers.append(question + ": " + sentences[index])
for answer in answers:
    print(answer + "\n")