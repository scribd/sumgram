from sumgram.sumgram_algorithm import get_args, set_log_defaults, set_logger_dets, get_top_sumgrams, print_top_ngrams
import spacy

nlp = spacy.load("en_core_web_sm")


def chunk_generator(document, chunk_size=200):
    length = len(document)
    for i in range(0, length, chunk_size):
        yield document[i:min(i + chunk_size, length)]


def get_docs_from_text(text, spacy_chunk_size, nlp):
    chunks = chunk_generator(text, chunk_size=spacy_chunk_size)
    list_of_docs = nlp.pipe(chunks)  # parallelizes creation of spacy docs.
    return list_of_docs


text = """Deep learning (also known as deep structured learning or differential programming) is part of a broader 
            family of machine learning methods based on artificial neural networks with representation learning. Learning 
            can be supervised, semi-supervised or unsupervised.[1][2][3] Deep learning architectures such as deep neural 
            networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied 
            to fields including computer vision, speech recognition, natural language processing, audio recognition, social 
            network filtering, machine translation, bioinformatics, drug design, medical image analysis, material inspection
             and board game programs, where they have produced results comparable to and in some cases surpassing human 
             expert performance. Artificial neural networks  were inspired by information processing and distributed 
             communication nodes in biological systems. Artificial neural networks have various differences from biological brains. Specifically, 
             neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic 
             (plastic) and analog."""

parser = get_args()

import os


class TextCleaningConfig:
    _config_dir, _ = os.path.split(__file__)
    _package_dir = os.path.normpath(_config_dir + os.sep + os.pardir)
    _data_dir = os.path.join(_package_dir, 'sumgram/sumgram/data/')
    _stopwords_folder = os.path.join(_data_dir, 'stopwords_lists/')
    _stopwords_file = 'stopwords_{language}.txt'
    STOPWORDS_FILE = _stopwords_folder + _stopwords_file



def get_stopwords_list(file):
    with open(file) as f:
        stopwords_list = [line.rstrip() for line in f]
        return stopwords_list


lan = "en"
args = parser.parse_args()
params = vars(args)
stopwords_file = TextCleaningConfig.STOPWORDS_FILE.format(language='en')
stopwords_list = get_stopwords_list(stopwords_file)
# params['no_rank_sentences'] = True
params['log_level'] = 'info'
params['min_df'] = 0.00001
params['add_stopwords'] = stopwords_list
params['no_pos_glue_split_ngrams'] = False
params['pos_glue_split_ngrams_coeff'] = 0.3
params['no_mvg_window_glue_split_ngrams'] = True
set_log_defaults(params)
set_logger_dets(params['log_dets'])

docs = get_docs_from_text(text, 2000, nlp)


def merge_all_chunks(nlp_docs):
    full_text = ""
    all_sentences = []
    for doc in nlp_docs:
        full_text += str(doc)
        all_sentences += format_sentences(doc)
    return {'text': full_text, 'sentences': all_sentences}


def format_sentences(nlp_doc):
    sentences = []

    def create_tok_obj(token):
        token_obj = {}
        token_obj['tok'] = token.text
        token_obj['pos'] = token.tag_
        token_obj['lemma'] = token.lemma_
        return token_obj

    def create_sent_obj(sent):
        sent_obj = {}
        sent_obj['sentence'] = str(sent)
        sent_obj['lemmatized_sentence'] = sent.lemma_
        for token in sent:
            sent_obj.setdefault('tokens', []).append(create_tok_obj(token))
        return sent_obj

    for sent in nlp_doc.sents:
        sentences.append(create_sent_obj(sent))

    return sentences


nlp_docs = [merge_all_chunks(docs)]

def proc_req(nlp_docs, params):
    params.setdefault('print_details', False)
    report = get_top_sumgrams(nlp_docs, params['base_ngram'], params)

    if ('top_sumgrams' in report and params['print_details'] is False):
        # since final top sumgrams not printed, print now
        print_top_ngrams(params['base_ngram'], report['top_sumgrams'], params['top_sumgram_count'],
                         params=report['params'])



proc_req(nlp_docs, params)
