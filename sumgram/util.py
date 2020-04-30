import gzip
import json
import logging
import os
import re
import sys
import tarfile

from subprocess import check_output, CalledProcessError
from multiprocessing import Pool

logger = logging.getLogger('sumGram.sumgram')


def genericErrorInfo(slug=''):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

    errMsg = fname + ', ' + str(exc_tb.tb_lineno) + ', ' + str(sys.exc_info())
    logger.error(errMsg + slug)

    return errMsg


def getStopwordsSet(frozenSetFlag=False):
    stopwords = getStopwordsDict()

    if (frozenSetFlag):
        return frozenset(stopwords.keys())
    else:
        return set(stopwords.keys())


def getStopwordsDict():
    stopwordsDict = {
        "a": True,
        "about": True,
        "above": True,
        "across": True,
        "after": True,
        "afterwards": True,
        "again": True,
        "against": True,
        "all": True,
        "almost": True,
        "alone": True,
        "along": True,
        "already": True,
        "also": True,
        "although": True,
        "always": True,
        "am": True,
        "among": True,
        "amongst": True,
        "amoungst": True,
        "amount": True,
        "an": True,
        "and": True,
        "another": True,
        "any": True,
        "anyhow": True,
        "anyone": True,
        "anything": True,
        "anyway": True,
        "anywhere": True,
        "are": True,
        "around": True,
        "as": True,
        "at": True,
        "back": True,
        "be": True,
        "became": True,
        "because": True,
        "become": True,
        "becomes": True,
        "becoming": True,
        "been": True,
        "before": True,
        "beforehand": True,
        "behind": True,
        "being": True,
        "below": True,
        "beside": True,
        "besides": True,
        "between": True,
        "beyond": True,
        "both": True,
        "but": True,
        "by": True,
        "can": True,
        "can\'t": True,
        "cannot": True,
        "cant": True,
        "co": True,
        "could not": True,
        "could": True,
        "couldn\'t": True,
        "couldnt": True,
        "de": True,
        "describe": True,
        "detail": True,
        "did": True,
        "do": True,
        "does": True,
        "doing": True,
        "done": True,
        "due": True,
        "during": True,
        "e.g": True,
        "e.g.": True,
        "e.g.,": True,
        "each": True,
        "eg": True,
        "either": True,
        "else": True,
        "elsewhere": True,
        "enough": True,
        "etc": True,
        "etc.": True,
        "even though": True,
        "ever": True,
        "every": True,
        "everyone": True,
        "everything": True,
        "everywhere": True,
        "except": True,
        "for": True,
        "former": True,
        "formerly": True,
        "from": True,
        "further": True,
        "get": True,
        "go": True,
        "had": True,
        "has not": True,
        "has": True,
        "hasn\'t": True,
        "hasnt": True,
        "have": True,
        "having": True,
        "he": True,
        "hence": True,
        "her": True,
        "here": True,
        "hereafter": True,
        "hereby": True,
        "herein": True,
        "hereupon": True,
        "hers": True,
        "herself": True,
        "him": True,
        "himself": True,
        "his": True,
        "how": True,
        "however": True,
        "i": True,
        "ie": True,
        "i.e": True,
        "i.e.": True,
        "if": True,
        "in": True,
        "inc": True,
        "inc.": True,
        "indeed": True,
        "into": True,
        "is": True,
        "it": True,
        "its": True,
        "it's": True,
        "itself": True,
        "just": True,
        "keep": True,
        "latter": True,
        "latterly": True,
        "less": True,
        "made": True,
        "make": True,
        "may": True,
        "me": True,
        "meanwhile": True,
        "might": True,
        "mine": True,
        "more": True,
        "moreover": True,
        "most": True,
        "mostly": True,
        "move": True,
        "must": True,
        "my": True,
        "myself": True,
        "namely": True,
        "neither": True,
        "never": True,
        "nevertheless": True,
        "next": True,
        "no": True,
        "nobody": True,
        "none": True,
        "noone": True,
        "nor": True,
        "not": True,
        "nothing": True,
        "now": True,
        "nowhere": True,
        "of": True,
        "off": True,
        "often": True,
        "on": True,
        "once": True,
        "one": True,
        "only": True,
        "onto": True,
        "or": True,
        "other": True,
        "others": True,
        "otherwise": True,
        "our": True,
        "ours": True,
        "ourselves": True,
        "out": True,
        "over": True,
        "own": True,
        "part": True,
        "per": True,
        "perhaps": True,
        "please": True,
        "put": True,
        "rather": True,
        "re": True,
        "same": True,
        "see": True,
        "seem": True,
        "seemed": True,
        "seeming": True,
        "seems": True,
        "several": True,
        "she": True,
        "should": True,
        "show": True,
        "side": True,
        "since": True,
        "sincere": True,
        "so": True,
        "some": True,
        "somehow": True,
        "someone": True,
        "something": True,
        "sometime": True,
        "sometimes": True,
        "somewhere": True,
        "still": True,
        "such": True,
        "take": True,
        "than": True,
        "that": True,
        "the": True,
        "their": True,
        "theirs": True,
        "them": True,
        "themselves": True,
        "then": True,
        "thence": True,
        "there": True,
        "thereafter": True,
        "thereby": True,
        "therefore": True,
        "therein": True,
        "thereupon": True,
        "these": True,
        "they": True,
        "this": True,
        "those": True,
        "though": True,
        "through": True,
        "throughout": True,
        "thru": True,
        "thus": True,
        "to": True,
        "together": True,
        "too": True,
        "toward": True,
        "towards": True,
        "un": True,
        "until": True,
        "upon": True,
        "us": True,
        "very": True,
        "via": True,
        "was": True,
        "we": True,
        "well": True,
        "were": True,
        "what": True,
        "whatever": True,
        "when": True,
        "whence": True,
        "whenever": True,
        "where": True,
        "whereafter": True,
        "whereas": True,
        "whereby": True,
        "wherein": True,
        "whereupon": True,
        "wherever": True,
        "whether": True,
        "which": True,
        "while": True,
        "whither": True,
        "who": True,
        "whoever": True,
        "whole": True,
        "whom": True,
        "whose": True,
        "why": True,
        "will": True,
        "with": True,
        "within": True,
        "without": True,
        "would": True,
        "yet": True,
        "you": True,
        "your": True,
        "yours": True,
        "yourself": True,
        "yourselves": True
    }

    return stopwordsDict


def sortDctByKey(dct, key, reverse=True):
    key = key.strip()
    if (len(dct) == 0 or len(key) == 0):
        return []

    return sorted(dct.items(), key=lambda x: x[1][key], reverse=reverse)


def dumpJsonToFile(outfilename, dictToWrite, indentFlag=True, extraParams=None):
    if (extraParams is None):
        extraParams = {}

    extraParams.setdefault('verbose', True)

    try:
        outfile = open(outfilename, 'w')

        if (indentFlag):
            json.dump(dictToWrite, outfile, ensure_ascii=False,
                      indent=4)  # by default, ensure_ascii=True, and this will cause  all non-ASCII characters in the output are escaped with \uXXXX sequences, and the result is a str instance consisting of ASCII characters only. Since in python 3 all strings are unicode by default, forcing ascii is unecessary
        else:
            json.dump(dictToWrite, outfile, ensure_ascii=False)

        outfile.close()

        if (extraParams['verbose']):
            logger.info('\twriteTextToFile(), wrote: ' + outfilename)
    except:
        genericErrorInfo('\n\terror: outfilename: ' + outfilename)


def getTextFromGZ(path):
    try:
        with gzip.open(path, 'rb') as f:
            return f.read().decode('utf-8')
    except:
        genericErrorInfo()

    return ''


def readTextFromTar(filename, addDetails=True):
    payload = []
    try:
        tar = tarfile.open(filename, 'r:*')

        for tarinfo in tar.getmembers():
            if tarinfo.isreg():

                try:
                    f = tar.extractfile(tarinfo)
                    text = f.read()

                    if (tarinfo.name.endswith('.gz')):
                        text = gzip.decompress(text)

                    text = text.decode('utf-8')
                    if (text != ''):
                        if (addDetails is True):
                            extra = {'src': filename}
                            text = getTextDetails(filename=os.path.basename(tarinfo.name), text=text, extra=extra)

                        payload.append(text)

                except UnicodeDecodeError as e:
                    logger.error('\nreadTextFromTar(), UnicodeDecodeError file: ' + tarinfo.name)
                except:
                    genericErrorInfo('\n\treadTextFromTar(), Error reading file: ' + tarinfo.name)

        tar.close()
    except:
        genericErrorInfo()

    return payload


def readTextFromFile(infilename):
    try:
        with open(infilename, 'r') as infile:
            return infile.read()
    except:
        genericErrorInfo('\n\treadTextFromFile(), error filename: ' + infilename)

    return ''


def getTextDetails(filename, text, extra=None):
    if (extra is None):
        extra = {}

    payload = {'filename': filename, 'text': text}

    for key, val in extra.items():
        payload[key] = val

    return payload


def readTextFromFilesRecursive(files, addDetails=True, curDepth=0, maxDepth=0):
    if (isinstance(files, str)):
        files = [files]

    if (isinstance(files, list) is False):
        return []

    if (maxDepth != 0 and curDepth > maxDepth):
        return []

    result = []
    for f in files:

        f = f.strip()

        if (f.endswith('.tar') or f.endswith('.tar.gz')):
            result += readTextFromTar(f, addDetails=addDetails)

        elif (f.endswith('.gz')):

            text = getTextFromGZ(f)
            if (text != ''):
                if (addDetails is True):
                    extra = {'depth': curDepth}
                    text = getTextDetails(filename=f, text=text, extra=extra)

                result.append(text)

        elif (os.path.isfile(f)):

            text = readTextFromFile(f)
            if (text != ''):
                if (addDetails is True):
                    extra = {'depth': curDepth}
                    text = getTextDetails(filename=f, text=text, extra=extra)

                result.append(text)

        elif (os.path.isdir(f)):

            if (f.endswith('/') is False):
                f = f + '/'

            secondLevelFiles = os.listdir(f)
            secondLevelFiles = [f + f2 for f2 in secondLevelFiles]
            result += readTextFromFilesRecursive(secondLevelFiles, addDetails=addDetails, curDepth=curDepth + 1,
                                                 maxDepth=maxDepth)

    return result


def change_format(nlp_docs):
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

    for sent in nlp_docs.sents:
        sentences.append(create_sent_obj(sent))

    return {'text': str(nlp_docs), 'sentences': sentences}


def overlapFor2Sets(firstSet, secondSet):
    intersection = float(len(firstSet & secondSet))
    minimum = min(len(firstSet), len(secondSet))

    if (minimum != 0):
        return round(intersection / minimum, 4)
    else:
        return 0


def parallelProxy(job):
    output = job['func'](**job['args'])

    if ('print' in job):
        if (len(job['print']) != 0):
            logger.info(job['print'])

    return {'input': job, 'output': output, 'misc': job['misc']}


def parallelTask(jobsLst, threadCount=5):
    if (len(jobsLst) == 0):
        return []

    if (threadCount < 2):
        threadCount = 2

    try:
        workers = Pool(threadCount)
        resLst = workers.map(parallelProxy, jobsLst)

        workers.close()
        workers.join()
    except:
        genericErrorInfo()
        return []

    return resLst


def phraseTokenizer(phrase):
    phrase = phrase.replace('\n', ' ')
    return re.split("[^a-zA-Z0-9.'’]", phrase)


def isMatchInOrder(keyLst, parentLst):
    if (len(keyLst) < 2 or len(parentLst) == 0):
        return False

    indices = []
    for key in keyLst:

        if (key not in parentLst):
            continue

        indx = parentLst.index(key)
        indices.append(indx)

    if (sorted(indices) == indices):
        return True
    else:
        return False


def rmStopwords(sent, stopwords):
    sent = sent.strip()
    if (sent == ''):
        return ''

    if (len(stopwords) == 0):
        return sent

    sent = sent.split(' ')
    newSent = []
    for tok in sent:

        if (tok.lower() in stopwords):
            continue

        newSent.append(tok)

    return newSent


def parallelGetTxt(folder, threadCount=5):
    folder = folder.strip()
    if (folder == ''):
        return []

    if (folder[-1] != '/'):
        folder = folder + '/'

    jobsLst = []
    files = os.listdir(folder)
    for i in range(len(files)):
        f = files[i].strip()

        keywords = {'infilename': folder + f}
        jobsLst.append({'func': readTextFromFile, 'args': keywords, 'misc': False})

    resLst = parallelTask(jobsLst, threadCount=threadCount)
    for res in resLst:
        res['text'] = res.pop('output')

        del res['input']['misc']
        del res['input']['func']
        del res['misc']

    return resLst


def sequentialGetTxt(folder):
    folder = folder.strip()
    if (folder == ''):
        return []

    if (folder[-1] != '/'):
        folder = folder + '/'

    resLst = []
    files = os.listdir(folder)

    for i in range(len(files)):
        resLst.append({
            'text': readTextFromFile(folder + files[i].strip())
        })

    return resLst


def getColorTxt(txt, ansiCode='91m'):
    return '\033[' + ansiCode + '{}\033[00m'.format(txt)
