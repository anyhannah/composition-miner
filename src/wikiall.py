from qwikidata.json_dump import WikidataJsonDump
from filelock import FileLock
import pandas as pd
import shutil
import sys
import os

CAP = 100000
SAVEDIR = 'nouns/wiki-all/'


def get_proc_num():
    with FileLock(os.path.join('temp', 'lock')):
        filenames = os.listdir('temp/')
        filenames.remove('lock')
        if len(filenames) == 0:
            new_num = 0
        else:
            current_num = filenames[-1]
            new_num = int(current_num) + CAP
            os.remove(os.path.join('temp', current_num))

        with open(os.path.join('temp', str(new_num)), 'w') as fp: pass

    return new_num


def main():
    # Obtain a csv file with Wikidata entities
    if len(sys.argv) < 2:
        raise ValueError("Please specify the compressed Wikidata file."
                         "Example: ``python3 src/wikiall.py latest-all.json.bz2``")
    elif len(sys.argv) > 2:
        raise ValueError('Too many arguments are given. Please specify the Wikidata file only.')
    wiki_compressed = sys.argv[1]

    # Create files for multiprocessing
    if not os.path.exists('temp/'):
        os.mkdir('temp/')
    if not os.path.exists(SAVEDIR):
        os.makedirs(SAVEDIR)
    with open(os.path.join('temp', 'lock'), 'w') as fp: pass

    # Run
    wjd = WikidataJsonDump(wiki_compressed)

    proc_num = get_proc_num()
    df = pd.DataFrame(columns=['id', 'label', 'parent', 'wikipedia', 'wordnet', 'synonym'])
    df.to_csv(os.path.join(SAVEDIR, "data-" + str(proc_num) + ".csv"), index=False)

    print(f'Extracting data into the ``{SAVEDIR}`` folder...')
    for ii, entity_dict in enumerate(wjd):
        if ii < proc_num:
            continue

        data = dict()
        data['id'] = entity_dict['id']
        if 'en' in entity_dict['labels'] and 'P279' in entity_dict['claims'].keys():
            # label
            data['label'] = entity_dict['labels']['en']['value']

            # subclass of
            data['parent'] = list()
            for subclass in entity_dict['claims']['P279']:
                if 'datavalue' in subclass['mainsnak'].keys():
                    data['parent'].append(subclass['mainsnak']['datavalue']['value']['id'])

            # Wikipedia
            data['wikipedia'] = ''
            if 'sitelinks' in entity_dict.keys() and 'enwiki' in entity_dict['sitelinks'].keys():
                data['wikipedia'] = entity_dict['sitelinks']['enwiki']['title']

            # WordNet 3.1
            data['wordnet'] = list()
            if 'P8814' in entity_dict['claims'].keys():
                for mainsnak_dict in entity_dict['claims']['P8814']:
                    if 'datavalue' in mainsnak_dict['mainsnak'].keys():
                        data['wordnet'].append(mainsnak_dict['mainsnak']['datavalue']['value'])

            # Exact match (WordNet 3.0 and wordNet 3.1)
            if 'P2888' in entity_dict['claims'].keys():
                for mainsnak_dict in entity_dict['claims']['P2888']:
                    if 'datavalue' in mainsnak_dict['mainsnak'].keys():
                        if 'http://wordnet-rdf.princeton.edu/' in mainsnak_dict['mainsnak']['datavalue']['value']:
                            data['wordnet'].append(mainsnak_dict['mainsnak']['datavalue']['value'].split("http://wordnet-rdf.princeton.edu/")[-1])

            # synonyms
            data['synonym'] = list()
            if 'en' in entity_dict['aliases'].keys():
                for synonym_dict in entity_dict['aliases']['en']:
                    data['synonym'].append(synonym_dict['value'])

            df = pd.DataFrame([data], columns=data.keys())
            df.to_csv(os.path.join(SAVEDIR, "data-" + str(proc_num) + ".csv"), mode='a', header=False, index=False)

        if ii == proc_num + CAP - 1:
            proc_num = get_proc_num()
            df = pd.DataFrame(columns=['id', 'label', 'parent', 'wikipedia', 'wordnet', 'synonym'])
            df.to_csv(os.path.join(SAVEDIR, "data-" + str(proc_num) + ".csv"), index=False)

    shutil.rmtree('temp/')


if __name__ == "__main__":
    main()