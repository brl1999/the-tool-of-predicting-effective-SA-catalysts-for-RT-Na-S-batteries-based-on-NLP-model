import re
import torch
from torch.nn import functional as F
import pandas as pd
from normalize_text import normalize

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
digits = "([0-9])"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def load_paper_to_sentences(path):
    # retrieve full paper
    with open(path, 'r') as f:
        sentences = f.readlines()
    sentences = ''.join(sentences)
    sentences = split_into_sentences(sentences)

    return sentences

def tokenize_sentences(sentences, tokenizer, max_length):
    norm_sents = [normalize(s) for s in sentences]
    tokenized_sents = tokenizer(norm_sents, padding='max_length', truncation=True, max_length=max_length)
    tokenized_sents = {k: torch.Tensor(v).long() for k, v in tokenized_sents.items()}
    return tokenized_sents

def get_naive_emb(tokenized_sents, model):
    with torch.no_grad():
        last_hidden_state = model(**tokenized_sents)[0]

    return torch.flatten(last_hidden_state.mean(dim=0))

def get_sum_hidden_emb(tokenized_sents, model):
    with torch.no_grad():
        hidden_states = model(**tokenized_sents, output_hidden_states=True).hidden_states
    stack_embs = torch.stack(hidden_states[-4:], dim=0) # use last 4 layers
    embs = torch.sum(stack_embs, dim=0)

    return torch.flatten(embs.mean(dim=0))

def get_similarities(embs_1, embs_2):
    similarities = {}
    for i, emb_1 in enumerate(embs_1):
        similarities[i] = []
        for emb_2 in embs_2:
            similarities[i].append(F.cosine_similarity(emb_1, emb_2, dim=0))

    return similarities

def get_pair_ranks(sim_list):
    sorted_id = sorted(range(len(sim_list)), key=lambda k: sim_list[k], reverse=True)

    return sorted_id

def load_csv_abstracts(csv_file):
    df = pd.read_csv(csv_file, names=['title', 'abs', 'pub', 'doi'])
    split_list = []
    for abs in df['abs']:
        split_list.append(split_into_sentences(abs))
    return df, split_list

def get_top_k(emb_list_aim, emb_list_compared, K):
    sim = get_similarities(emb_list_aim, emb_list_compared)
    topk = {}
    for key in sim.keys():
        topk[key] = get_pair_ranks(sim[key])[:K]
    return topk