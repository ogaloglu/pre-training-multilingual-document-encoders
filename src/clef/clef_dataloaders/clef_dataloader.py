import os
import sys
import codecs
import urllib
from lxml import html as etree
# TODO: change
sys.path.insert(0, '/home/ogalolu/thesis/pre-training-multilingual-document-encoders/clef/clef_dataloaders')
from clef_paths import get_lang2pair, PATH_BASE_QUERIES, PATH_BASE_EVAL, CLEF_LOWRES_DIR, all_paths
from typing import Tuple, List


def _decode_xml(path, encoding):
    if encoding == "plain":
        with open(path, "r") as f:
            xml = f.read()
    else:
        with codecs.open(path, encoding=encoding) as f:
            xml = f.read()
        
    xml = xml.replace("<BODY>", "<BODYY>").replace("</BODY>", "</BODYY>")
    xml = xml.replace("<HEAD>", "<HEADD>").replace("</HEAD>", "</HEADD>")
    xml = "<root>" + xml + "</root>"
    xml = xml.replace("<IZV>", "").replace("</IZV>", "")  # for russian corpus
    return etree.fromstring(xml)


def load_relevance_assessments(language: str, year: str, load_non_relevant_docs=False):
    assert year in ["2001", "2002", "2003"]
    doc_lang_full = get_lang2pair(language)[1]
    path = os.path.join(PATH_BASE_EVAL + year, "qrels_" + doc_lang_full)
    positive_list = {}
    negative_list = {}
    with open(path) as f:
        for line in f.readlines():
            tokens = line.rstrip("\n").split(" ")
            # check if document is relevant for query
            relevance = int(tokens[len(tokens) - 1])
            if relevance != 0:
                query_id = int(tokens[0].strip())
                document_id = tokens[2].strip()
                if query_id not in positive_list:
                    relevant_docs = [document_id]
                else:
                    relevant_docs = positive_list[query_id]
                    relevant_docs.append(document_id)
                positive_list[query_id] = relevant_docs
              
            if load_non_relevant_docs and relevance == 0:
                query_id = int(tokens[0].strip())
                document_id = tokens[2].strip()
                if query_id not in negative_list:
                    non_relevant_docs = [document_id]
                else:
                    non_relevant_docs = negative_list[query_id]
                    non_relevant_docs.append(document_id)
                negative_list[query_id] = non_relevant_docs
          
    result = negative_list if load_non_relevant_docs else positive_list
    return result

_docs_cache = {}
def load_documents(language, year, limit_documents=None, only_body=False, **kwargs):
    """
    Walks through document files and extracts content.
    :param language: corpus language
    :param year: year of CLEF campaign (2001-2003)
    :param limit_documents: load only top k documents
    :param only_body: used for masked language modelling corpus
    :return:
    """
    def _load_file(file_path, extractor, encoding, limit=None, only_body=False, **kwargs):
        tree = _decode_xml(file_path, encoding)
        docs = []
        ids = []
        rerank_set = kwargs.get("rerank_corpus", None)
        for i, doc in enumerate(list(tree)):
            if len(docs) == limit:
                break
            document_id, full_text = extractor(doc, only_body=only_body)
            if document_id is not None and full_text is not None:
                if rerank_set and document_id not in rerank_set:
                    continue
                ids.append(document_id)
                docs.append(full_text)
        return ids, docs
    
    doc_lang_iso, doc_lang_full = get_lang2pair(language)
    key = f"{doc_lang_iso}_{limit_documents}"
    
    # Always load corpus if we do re-ranking.
    do_reranking = bool(kwargs.get("rerank_corpus", False))
    corpus_is_in_memory = key in _docs_cache
    
    if not corpus_is_in_memory or do_reranking:
        doc_dirs = all_paths[doc_lang_iso][year]
        documents = []
        doc_ids = []
        limit_reached = False
        encoding = 'UTF-8' if doc_lang_iso == "ru" or doc_lang_full == "russian" else 'ISO-8859-1'
        for doc_dir, extractor in doc_dirs:
            if not limit_reached:
                for file in next(os.walk(doc_dir))[2]:
                    if not file.endswith(".dtd"):
                        tmp_doc_ids, tmp_documents = _load_file(
                            file_path = doc_dir + file,
                            encoding = encoding,
                            extractor=extractor,
                            limit_documents=limit_documents,
                            only_body=only_body, **kwargs
                        )
                        documents.extend(tmp_documents)
                        doc_ids.extend(tmp_doc_ids)
                    if limit_documents and len(documents) >= limit_documents:
                        limit_reached = True
                        break
        if not do_reranking:
            _docs_cache[key] = (doc_ids, documents)
    else:
        doc_ids, documents = _docs_cache[key]
      
    return doc_ids, documents


def load_queries(language, year, limit=None, encoding=None, include_desc=True) -> Tuple[List, List]:
    language = language.lower()
    
    if language == "sw" or language == "swahili":
        return _load_swahili_queries(year)
    
    if language == "so" or language == "somali":
        return _load_somali_queries(year)
    
    if year == "2000":
        return _load_clef2000_queries(language)
    
    lang_iso, lang_full = get_lang2pair(language)
    path = os.path.join(PATH_BASE_QUERIES + year, "Top-" + lang_iso + year[-2:] + ".txt")
    
    if not encoding:
        encoding = 'UTF-8' if lang_iso == "ru" or lang_full == "russian" in path else 'ISO-8859-1'
      
    tree = _decode_xml(path, encoding)
    tag_title = lang_iso + '-title'
    tag_desc = lang_iso + '-desc'
    # tag_narr = language_tag + '-narr'
    
    queries = []
    ids = []
    for i, topic in enumerate(list(tree)):
        _id = topic.findtext('num').strip() # e.g. 'C041'
        _id = int(_id[1:]) # e.g. 41
        title = topic.findtext(tag_title)
        if include_desc:
            desc = topic.findtext(tag_desc)
            query = ' '.join([title, desc])
        else:
            query = title
        queries.append(query)
        ids.append(_id)
        if i == limit:
            break
        
    # newly added:
    # queries = [q.replace("\n", " ").replace("\r", " ") for q in queries]
    
    return ids, queries


def load_clef(query_lang: str, doc_lang: str, year: str = "2003", limit_documents=None, **kwargs):
    year = str(year)
    assert year in ["2001", "2002", "2003", "all"]
    
    query_lang_iso, query_lang_full = get_lang2pair(query_lang)
    
    # Load relevance assessments
    relass = load_relevance_assessments(doc_lang, year=year)
    
    # Load queries
    query_ids, queries = load_queries(
        language=query_lang_iso,
        year=year,
        limit=None,
        encoding="plain" if query_lang_iso == "ru" else None,
        include_desc=kwargs.get("include_desc", True)
    )
    
    # Load documents
    doc_ids, documents = load_documents(language=doc_lang, year=year, limit_documents=limit_documents, **kwargs)
    return doc_ids, documents, query_ids, queries, relass


def load_clef_rerank(dlang: str, qlang: str, rerank_dir: str, topk: int, include_desc: bool=True):
    """
    For efficiency, we only load a document if it's within the top-k of any query.
    :param dlang: ISO-code of document language
    :param qlang: ISO-code of query language
    :param rerank_dir: directory containing one file for each query (filename = query-id), content: list of document-ids (=pre-ranking).
    :param topk: cut-off, number of documents to re-rank for each query
    :param include_desc: clef description 
    :return: document ids, documents, queries, query ids, relevance assessments, list of top-k document ids for each query (=input to be re-ranked)
    """
    rerank_corpus = set()
    qid2topk_rerank = {}
    tmp = os.listdir(rerank_dir)
    for elem in tmp:
        qid = int(elem.split("/")[-1][:-4])
        rerank_doc_ids = []
        with open(rerank_dir + elem, "r") as f:
            for i, did_score in enumerate(f):
                did = did_score.split("\t")[0]
                did = did.strip()
                rerank_doc_ids.append(did)
                if i < topk:
                    rerank_corpus.add(did)
        qid2topk_rerank[qid] = rerank_doc_ids
      
    doc_ids, documents, query_ids, queries, relass = load_clef(
        query_lang=qlang,
        doc_lang=dlang,
        rerank_corpus=rerank_corpus,
        include_desc=include_desc)
    return doc_ids, documents, queries, query_ids, relass, qid2topk_rerank


def _load_clef2000_queries(language):
    short2lang = {"en": "E", "fr": "F", "fi": "FI", "de": "D", "it": "I", "sw": "SW", "es": "SP"}
    long2lang = {"engnlish": "E", "french": "F", "finnish": "FI", "german": "D", "italian": "I", "swedish": "SW", "spanish": "SP"}
    if language in short2lang:
        lang = short2lang[language]
    elif language in long2lang:
        lang = long2lang[language]
    else:
        raise NotImplementedError
    
    encoding = 'UTF-8' if lang == "ru" or lang == "russian" else 'ISO-8859-1'
    with codecs.open(os.path.join(PATH_BASE_QUERIES, "topics2000", f"TOP-{lang}.txt"), encoding=encoding) as f:
        lines = f.readlines()
      
    topics = []
    topic = {}
    lines = iter(lines)
    for line in lines:
        line = line.strip()
        if line == "":
            continue
            
        elif line.startswith("<num>"):
            assert 'num' not in topic
            topic['num'] = line.replace("<num>", "").strip()

        elif line.startswith(f"<{lang}-desc>"):
            desc = []
            line = next(lines).strip()
            while line != "":
                desc.append(line)
                line = next(lines).strip()
            assert 'desc' not in topic
            topic['desc'] = " ".join(desc)
            
        elif line.startswith(f"<{lang}-title>"):
            line = next(lines).strip()
            assert 'title' not in topic
            topic['title'] = line
            
        elif line.startswith(f"<{lang}-narr>"):
            narr = []
            line = next(lines).strip()
            while line != "</top>":
                if line:
                    narr.append(line)
                line = next(lines).strip()
            assert 'narr' not in topic
            topic['narr'] = " ".join(narr)

        else:
            assert line.strip() in ["<top>", "</top>"]
            if topic:
                assert all([tag in topic for tag in ['num', 'title', 'desc', 'narr']])
                topics.append(topic)
            topic = {}
        
    if topic: topics.append(topic)
    
    queries = []
    qids = []
    for topic in topics:
        qids.append(int(topic['num'][1:]))
        queries.append(f"{topic['title']} {topic['desc']}")
    return qids, queries


def _load_lowres_queries(filepath: str, year: str):
    if not os.path.exists(filepath):
        url = "https://ciir.cs.umass.edu/downloads/ictir19_simulate_low_resource/ictir19_simulate_low_resource.zip"
        raise FileNotFoundError(f"Please download\n\n{url}\n\nand set CLEF_LOWRES_DIR in paths.py")
    
    with open(filepath, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    qids, _queries = [], []
    
    # CLEF query ids
    if year == "2001":
        # 2001: 41-90
        is_valid_query_id = lambda query_id: 0 <= query_id <= 90
    elif year == "2002":
        # 2002: 91-140
        is_valid_query_id = lambda query_id: 91 <= query_id <= 140
    else:
        # 2003: 141-200
        assert year == "2003"
        is_valid_query_id = lambda query_id: 141 <= query_id <= 200
      
    while lines:
        qid_line = lines.pop(0)
        qid = qid_line.split("query_id:")[-1].strip()
        qid = int(qid[1:])
        
        title_line = lines.pop(0)
        title = title_line.split("title:")[-1].strip()
        
        desc_line = lines.pop(0)
        desc = desc_line.split("description:")[-1]
        
        while lines and lines[0] != "":
            desc_line = lines.pop(0)
            desc_p2 = desc_line.split("description:")[-1]
            desc = f"{desc} {desc_p2}"
          
        while lines and lines[0] == "":
            lines.pop(0)
          
        if is_valid_query_id(qid):
            qids.append(qid)
            _queries.append(f"{title} {desc}")
        
    return qids, _queries


def _load_swahili_queries(year: str):
    return _load_lowres_queries(
        os.path.join(CLEF_LOWRES_DIR, "clef-en-2000-2003-wo-narrative-Day-2_SWAHILI.txt"),
        year
    )


def _load_somali_queries(year: str):
    return _load_lowres_queries(
        os.path.join(CLEF_LOWRES_DIR, "clef-en-2000-2003-wo-narrative-Day-2_SOMALI.txt"),
        year
    )
    