import pandas as pd
from os import listdir
from os.path import isfile, join
import wikipedia
import warnings
import re
import csv
import tqdm
import benepar
import nltk
benepar.download('benepar_en3')
nltk.download('wordnet')
nltk.download('wordnet31')
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet31 as wn31

DIRPATH = 'nouns/wiki-all/'
if not DIRPATH.endswith('/'):
    DIRPATH += '/'
RESULT_FILE = 'nouns/filtered_wikidata.csv'

exclude_keywords = {
    'abstract data type', 'administrative division', 'algebra', 'algebraic geometry', 'alphabet', 'annotation',
    'BDSM', 'bishop', 'broadcast',
    'category theory', 'chess', 'chess composition', 'color', 'colour', 'computer programming', 'computer science',
    'computer security', 'computing', 'control theory', 'cooking technique', 'cryptography', 'currency', 'curve',
    'dance', 'databases', 'data mining', 'digital', 'drama', 'Dungeons & Dragons', 'economics',
    'fiction', 'formal languages', 'feudal barony',
    'gaming', 'gay culture', 'gender', 'genre', 'geometry', 'graph theory', 'graphic design', 'group', 'group theory',
    'hairstyle', 'heraldry',
    'image processing', 'information', 'international law', 'law', 'linear algebra', 'literature', 'literary theory',
    'management', 'material', 'mathematical logic','mathematics', 'mathematics and physics',
    'measure theory', 'memory management', 'mineralogy', 'Mozilla',
    'narrative', 'number theory',
    'optics', 'order theory', 'pattern', 'physics', 'Pictish symbol', 'polygon', 'predicate logic', 'programming',
    'protein',
    'representation theory', 'ring theory', 'road tax',
    'saying', 'set theory', 'settlement or colony', 'sexuality', 'social and political', 'sociology', 'software',
    'software development', 'SQR', 'Star Trek', 'Star Wars', 'statistics', 'stereotype', 'system call',
    'systems architecture',
    'technology', 'telecommunications', 'topology', 'unit', 'video games'
}

hard_exclude_wiki_under = {
    'Q4936952': 'anatomical structure',
    'Q151885': 'concept',
    'Q20742825': 'cultural depiction',
    'Q5720030': 'detection',
    'Q778379': 'expression',
    'Q2095': 'food',
    'Q649732': 'formal system',
    'Q11410': 'game',
    'Q483394': 'genre',
    'Q16829513': 'material',
    'Q12140': 'medication',
    'Q10301427': 'moving image',
    'Q2031291': 'musical release',
    'Q1318295': 'narrative',
    'Q64728693': 'non-existent entity',
    'Q174211': 'organic compound',
    'Q7239': 'organism',
    'Q43229': 'organization',
    'Q853520': 'paratext',
    'Q4303335': 'program',
    'Q4393498': 'representation',
    'Q11461': 'sound',
    'Q17451': 'typeface'
}

soft_exclude_wiki_under = {
    'Q538722': 'applied computer science',
    'Q11028': 'information',
    'Q3833047': 'link',
    'Q3249551': 'process',
    'Q7397': 'software',
    'Q17176533': 'software component',
    'Q1900326': 'network',
}


exclude_headnouns = {'area', 'areas', 'art', 'arts', 'base', 'body', 'business',
                     'capability', 'cemetery', 'city', 'cities', 'community',
                     'establishment', 'facility', 'facilities', 'garden', 'hall', 'housing',
                     'installation', 'land', 'lands', 'lane', 'lanes', 'location', 'locations',
                     'mechanism', 'mineral', 'minerals',
                     'path', 'passage', 'place', 'places',
                     'settlement', 'space', 'station', 'stations', 'street', 'thoroughfare', 'town', 'track',
                     'remains', 'residence', 'road', 'roads', 'roadway', 'room', 'rooms',
                     'venue', 'venues', 'way', 'work', 'workplace'}


def wiki_parents():
    # For each of the Wikidata items, get its parents
    file_list = [f for f in listdir(DIRPATH) if isfile(join(DIRPATH, f)) and f.endswith('.csv')]
    superclass_dict = dict()

    for filename in file_list:
        wiki_df = pd.read_csv(DIRPATH + filename, encoding='utf-8')
        for index, row in wiki_df.iterrows():
            if row['parent'] == "[]":
                superclass_dict[row['id']] = ''
            else:
                superclass_dict[row['id']] = list()
                for parent in row['parent'].replace('[', '').replace(']', '').replace("'", "").split(", "):
                    superclass_dict[row['id']].append(parent)
    return superclass_dict


def all_ancestors(child, trace, superclass_dict):
    # Return the ancestors of the given child
    if (child not in superclass_dict.keys()) or (child in trace):
        return set()
    parent_list = superclass_dict[child]
    if not parent_list:
        return set()
    trace.add(child)
    return set(parent_list + [ancestor for parent in parent_list
                              for ancestor in all_ancestors(parent, trace, superclass_dict)])


def get_synset_lemma(wn_version):
    synset_offsets = set()
    lemmas = set()
    for synset in wn_version.all_synsets('n'):
        synset_offsets.add(str(synset.offset()).zfill(8) + '-n')
        lemmas1 = [l.name().replace("_", " ") for l in synset.lemmas()]
        lemmas2 = [l.replace("-", " ") for l in lemmas1]
        lemmas = lemmas | set(lemmas1) | set(lemmas2)
    return synset_offsets, lemmas


def get_first_sentence(text):
    # split the given text into sentences and return the first one
    alphabets = "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    digits = "([0-9])"
    multiple_dots = r'\.{2,}'

    text = re.sub(r' \([^)]*\)', '', text)  # remove parentheses and their content

    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = text.replace(".)", "<prd>)")

    unclosed_paren = re.findall("[(](.*?)[.]", text)
    while unclosed_paren and (unclosed_paren[0].count(")") - unclosed_paren[0].count("(") != 1):
        text = re.sub("[(](.*?)[.]", "(\\1<prd>", text)
        unclosed_paren = re.findall("[(](.*?)[.]", text)
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    if " cf." in text: text = text.replace(" cf.", " cf<prd>")
    if " lit." in text: text = text.replace(" lit.", " lit<prd>")
    if "approx." in text: text = text.replace("approx.", "approx<prd>")
    if "orig." in text: text = text.replace("orig.", "orig<prd>")

    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")

    text = re.sub("([.]+)", "\\1<stop>", text)
    text = re.sub("([?]+)", "\\1<stop>", text)
    text = re.sub("(!+)", "\\1<stop>", text)
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]

    return sentences[0]


def extract_nn(tree):
    while any(node.label() in {'NP', 'NML'} for node in tree):
        for i in range(len(tree)):
            node = tree[i]
            if node.label() in {'NP', 'NML'}:
                if node.leaves()[-1] in {'any', 'class', 'classes', 'cluster', 'clusters',
                                         'form', 'forms', 'group', 'groups',
                                         'instance', 'instances', 'kind', 'kinds',
                                         'one', 'ones', 'name', 'names', 'number',
                                         'pair', 'pairs', 'parcel', 'parcels', 'part', 'parts',
                                         'patch', 'patches', 'piece', 'pieces',
                                         'set', 'sets',
                                         'term', 'terms', 'type', 'types',
                                         'variety', 'varieties'} and\
                        i != (len(tree)-1) and tree[i+1].label() in {'PP', 'IN'}:
                    if tree[i+1].label() == 'PP':
                        if tree[i+1][0].label() == 'IN':
                            tree = tree[i+1]
                        else:
                            tree = tree[i+1][0]
                    else: # IN
                        tree = tree[i+2]
                else:
                    tree = node
                    # first NP contains the head noun of the phrase
                    break
    for i in reversed(range(len(tree))):
        # the last noun is the head noun
        if tree[i].label() in {'NN', 'NNS'}:
            return tree[i].leaves()[0]
    return ""


def find_head_noun(sent, parser):
    try:
        tree = parser.parse(sent)
    except ValueError:
        # sentence too long
        return ""

    if tree.label() == 'TOP':
        tree = tree[0]

    try:
        labels = [node.label() for node in tree]
        while 'VP' not in labels:
            # heuristics assuming the main content would appear under the first S
            tree = tree[0]
            labels = [node.label() for node in tree]

        while 'VP' in labels:
            tree = tree[labels.index('VP')]
            labels = [node.label() for node in tree]
        if tree[0].leaves() == ['is'] or tree[0].leaves() == ['are'] or \
                tree[0].leaves() == ['was'] or tree[0].leaves() == ['were'] or\
                tree[0].leaves() == ['denotes'] or tree[0].leaves() == ['denote']:
            head_noun = extract_nn(tree)
            if head_noun.strip():
                return head_noun
            else:  # does not have the form 'be+NP'
                return ""
        elif " ".join(tree.leaves()[:2]) == 'refers to' or \
                " ".join(tree.leaves()[:2]) == 'refer to':
            for node in tree:
                if node.label() == 'PP':
                    head_noun = extract_nn(node[1])
                    return head_noun
        else:
            return ""
    except (IndexError, AttributeError):
        return ""


def filter_wikidata(superclass_dict, parser, synset_wn30_set, synset_wn31_set, lemma_wn_set):
    """
    Generates a csv file which includes things under
    'artificial physical object',  'artificial physical structure', or 'artificial entity'
    and excludes certain keywords from ``exclude_keywords``
    and excludes things under ``exclude_wiki_under``, e.g. process(Q3249551), abstract object(Q11028)
    """

    filtered_wiki_dict = dict()
    filtered_wiki_dict['id'] = list()
    filtered_wiki_dict['label'] = list()
    filtered_wiki_dict['wikipedia'] = list()
    filtered_wiki_dict['redirect'] = list()
    filtered_wiki_dict['wordnet'] = list()
    filtered_wiki_dict['synonym'] = list()

    file_list = [f for f in listdir(DIRPATH) if isfile(join(DIRPATH, f)) and f.endswith('.csv')]

    with open(RESULT_FILE, 'w') as wiki_f:
        writer = csv.writer(wiki_f, delimiter='\t')
        writer.writerow(filtered_wiki_dict.keys())

        for filename in tqdm.tqdm(file_list):
            wiki_df = pd.read_csv(DIRPATH+filename, encoding='utf-8').fillna('')

            for index, row in wiki_df.iterrows():
                # filter out by keywords from their Wikipedia article names or from the hierarchy data
                if pd.isnull(row['wikipedia']) or \
                        any('('+keyword+')' in str(row['wikipedia']) for keyword in exclude_keywords) or \
                        (row['id'] in hard_exclude_wiki_under.keys()) or (row['id'] in soft_exclude_wiki_under.keys()):
                    continue
                wiki_id = row['id']
                entry = str(row['label'])
                wikipedia_title = str(row['wikipedia'])
                wordnet_link = row['wordnet'].replace("[", "").replace("]", "").replace("'", "")
                synonyms = row['synonym'].replace("[", "").replace("]", "").replace("'", "")
                # synonyms help detecting words like bicycle saddle, swim fins, etc.

                # Pick nouns that exist in WordNet
                if entry in lemma_wn_set or \
                        any(any(synonym_var in lemma_wn_set for synonym_var in
                                [synonym, synonym.replace(" ", ""), synonym.replace(" ", "-"), synonym.rstrip('\s'),
                                 synonym.rstrip('\s').rstrip('\e')]) for synonym in synonyms.split(", ")) or \
                        any(('wn30/' in link and link.split("/")[-1] in synset_wn30_set) for link in
                            wordnet_link.split(", ")) or \
                        any(('wn30/' not in link and link.split("/")[-1] in synset_wn31_set) for link in
                            wordnet_link.split(", ")):

                    wikipedia_strip = wikipedia_title.split("(")[0].strip()
                    # Filtering out nouns based on their Wikipedia article title would be more accurate
                    # if we use their redirected title names, but to reduce the number of times to
                    # call ``wikipedia.page`` function (because of the time constraint),
                    # the filter is applied to ``wikipedia_title`` instead of ``wikipedia_redirect``
                    if any(entry_letter.isdigit() for entry_letter in entry + wikipedia_strip) or \
                            ('(' in wikipedia_title and any(
                                ch.isupper() for ch in wikipedia_title.split("(")[1])) or \
                            (' ' in wikipedia_strip and
                             any(wiki_token[0].isupper() for wiki_token in
                                 wikipedia_strip.split(" ", 1)[-1].split(" "))):
                        # 1. If the wikidata entry contains a digit, the type is probably too specific.
                        # (same for wikipedia entry; e.g., Lamborghini 350GTV)
                        # 2. If there is an uppercase letter within the parentheses of the wikipedia entry,
                        # that's probably a country-specific or other named entities.
                        # However, synonyms that contain digits are fine; e.g., fire extinguisher = DIN EN 3
                        # 3. If there is a space-separated token (that is not at the beginning of the word) that
                        # starts with an uppercase letter, then, that's likely to be a named entity.
                        # 4. The head noun of the first sentence of its wikipedia article is in the pre-defined
                        # list to exclude -- mostly for structures/places without parts
                        continue

                    ancestors = set(all_ancestors(wiki_id, set(), superclass_dict))

                    # Only interested in nouns under artificial object (Q16686448) with exceptions.
                    # This should cover nouns under artificial physical object (Q8205328),
                    # artificial physical structure (Q11908691), etc.
                    # Exceptions are any nouns under the entities in 'hard_exclude_wiki_under'
                    # and some of the nouns under the entities in 'soft_exclude_wiki_under', except:
                    # Q2858615: electronic machine
                    # Q32914898: portable object
                    # Q10273457: equipment
                    # Q1485500: tangible good
                    # Q122853586: physical component
                    if ('Q16686448' in ancestors) and not (hard_exclude_wiki_under.keys() & ancestors) and \
                            (not (soft_exclude_wiki_under.keys() & ancestors) or \
                             ({'Q2858615', 'Q32914898', 'Q10273457', 'Q1485500', 'Q122853586'} & ancestors)):
                        try:
                            wikipedia_page = wikipedia.page(wikipedia_title, auto_suggest=False)
                        # except wikipedia.exceptions.PageError:
                        except:
                            try:
                                wikipedia_page = wikipedia.page(wikipedia_title)
                            except:
                                # ambiguous names may refer to multiple articles but what they refer to is unclear
                                continue
                        wikipedia_redirect = wikipedia_page.title

                        wiki_content = wikipedia_page.content
                        if not wiki_content.strip():
                            continue

                        sent = get_first_sentence(wiki_content)
                        head_noun = find_head_noun(sent, parser)
                        if head_noun in exclude_headnouns:
                            continue
                        # For nouns whose Wikipedia article's head noun is "building", filter out
                        # 1. multiword entries (airport terminal, convention center, etc.),
                        # 2. compound words (boathouse, courthouse, skyscraper, etc.), and
                        # 3. words longer than 6 letters
                        if head_noun == 'building':
                            if ' ' in wikipedia_strip:
                                continue
                            for i in range(1, len(wikipedia_strip) - 2):
                                if wikipedia_strip.lower()[:i + 1] in set([str(lemma.name())
                                                                           for synset in
                                                                           wn.synsets(wikipedia_strip[:i + 1])
                                                                           for lemma in synset.lemmas()]) and \
                                        wikipedia_strip.lower()[i + 1:] in set([str(lemma.name())
                                                                                for synset in
                                                                                wn.synsets(wikipedia_strip[i + 1:])
                                                                                for lemma in synset.lemmas()]):
                                    continue
                            if len(wikipedia_strip) > 6:
                                continue

                        # If none of the filters applies to the entry, save the noun.
                        writer.writerow([wiki_id, entry, wikipedia_title, wikipedia_redirect, wordnet_link, synonyms])


def main():
    warnings.filterwarnings("ignore")

    print("Preparing reference data...")
    superclass_dict = wiki_parents()
    synset_wn30_set, lemma_wn30_set = get_synset_lemma(wn)
    synset_wn31_set, lemma_wn31_set = get_synset_lemma(wn31)
    parser = benepar.Parser("benepar_en3")

    print("Getting filtered entries from Wikidata...")
    filter_wikidata(superclass_dict, parser, synset_wn30_set, synset_wn31_set, lemma_wn30_set | lemma_wn31_set)


if __name__ == "__main__":
    main()