import openai
from retry import retry
import pandas as pd
import re
import sys
import os

PROMPT_FOLDER = 'nouns/filtering-nouns/'
if not PROMPT_FOLDER.endswith('/'):
    PROMPT_FOLDER += '/'
SAVE_FILE = 'nouns/nounlist.txt'

OPENAI_API_KEY = '[API_KEY]'


def reformat_last_file(folder, filename, file_number):
    # Reformat the last file
    with open(folder+filename+file_number+'.txt', 'r+') as f:
        content = f.read()
        # assume the only lines starting with digits are the list of entities
        num_nouns = len([line.split(".", 1)[0] for line in content.split("\n") if line and line[0].isdigit()])
        # change 100 nouns to a proper number
        content = content.replace('the following 50 ', 'the following ' + str(num_nouns) + ' ')
        content = content.replace('the following 1 things', 'the following 1 thing')
        content = content.replace('the following 1 nouns', 'the following 1 noun')
        content = content.strip()  # remove a new line at the end
        f.seek(0)
        f.write(content)
        f.truncate()


def write_easy_prompts(folder, filename, wiki_df):
    # This function avoids duplicates of Wikipedia articles
    count = 0
    all_nouns = set()
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Write prompts to get easy (well-perceived) nouns
    for index, row in wiki_df.iterrows():
        file_number = str(int(count / 50))
        if len(file_number) == 1:
            file_number = '0' + file_number

        with open(folder+filename+file_number+'.txt', 'a') as f:
            label = row['redirect']
            if label and (label not in all_nouns):
                # In some cases, multiple Wikidata entries are mapped to the same Wikipedia articles.
                all_nouns.add(label)
                if count % 50 == 0:
                    f.write("How likely are the following 50 things to be commonly recognized by a typical sixth-grader? "
                            "Add \" - [likely / probably likely / probably unlikely / unlikely] "
                            "to be recognized by sixth-graders\" after the nouns in the list. "
                            "Please do not alter the names within parentheses.\n\n")
                f.write(str(count % 50 + 1) + ". " + label)
                if (count % 50) != 49:
                    f.write("\n")
                count += 1

    # Last modified file name and number
    reformat_last_file(folder, filename, file_number)


def write_phys_prompts(response_folder, prompt_folder, prompt_filename):
    count = 0
    if not os.path.exists(prompt_folder):
        os.makedirs(prompt_folder)

    # Write prompts to get physical entities
    for response_filename in sorted(os.listdir(response_folder)):
        if response_filename.endswith('.txt'):
            with open(response_folder+response_filename, 'r') as f_read:
                line = f_read.readline()
                while not re.match("^[0-9]+\.", line):
                    line = f_read.readline()
                while re.match("^[0-9]+\.", line):
                    if 'unlikely' not in line.lower():
                        file_number = str(int(count/50))
                        if len(str(int(count/50))) == 1:
                            file_number = '0'+file_number
                        with open(prompt_folder+prompt_filename+file_number+'.txt', 'a') as f_write:
                            if count%50 == 0:
                                f_write.write("Could you classify the following 50 nouns based on whether they "
                                              "primarily refer to standalone physical objects, standalone built "
                                              "structures, substances, or neither? Add \" - is a [physical object / "
                                              "built structure / substance / neither]\" after the nouns in the list. "
                                              "Please do not alter the names within parentheses.\n\n"
                                              "Here are the criteria for each category:\n"
                                              "- Physical objects: Tangible items that can exist independently, "
                                              "or items that might be part of a larger entity but can be replaced.\n"
                                              "- Built structures: Man-made constructions that serve as "
                                              "physical places or infrastructure.\n"
                                              "- Substances: Any substance with uniform characteristics or "
                                              "any matter that can be best characterized by their chemical composition.\n"
                                              "- Neither: None of the above.\n\n")
                            f_write.write(str(count % 50 + 1)+". " + line.split(".", 1)[1].rsplit("- ", 1)[0].rsplit(" -", 1)[0].strip())
                            if count % 50 != 49:
                                f_write.write("\n")
                        count += 1
                    line = f_read.readline()

    reformat_last_file(prompt_folder, prompt_filename, file_number)


def write_count_prompts(response_folder, prompt_folder, prompt_filename):
    count = 0
    if not os.path.exists(prompt_folder):
        os.makedirs(prompt_folder)

    # Write prompts to get countable entities
    for response_filename in sorted(os.listdir(response_folder)):
        if response_filename.endswith('.txt'):
            with open(response_folder + response_filename, 'r') as f_read:
                line = f_read.readline()
                while not re.match("^[0-9]+\.", line):
                    line = f_read.readline()
                while re.match("^[0-9]+\.", line):
                    if not ('neither' in line.lower() or 'substance' in line.lower()):
                        file_number = str(int(count / 50))
                        if len(str(int(count / 50))) == 1:
                            file_number = '0' + file_number
                        with open(prompt_folder + prompt_filename + file_number + '.txt', 'a') as f_write:
                            if count % 50 == 0:
                                f_write.write("Could you classify the following 50 nouns based on whether they are "
                                              "in general used as a mass noun or a count noun? Add \" - mass noun\" "
                                              "or \" - count noun\" accordingly after the nouns in the list.\n\n")
                            f_write.write(str(count % 50 + 1) + ". " +
                                          line.split(".", 1)[1].rsplit("- ", 1)[0].rsplit(" -", 1)[0].strip())
                            if count % 50 != 49:
                                f_write.write("\n")
                        count += 1
                    line = f_read.readline()

    reformat_last_file(prompt_folder, prompt_filename, file_number)


def write_indiv_prompts(response_folder, prompt_folder, prompt_filename):
    # Filters out an item with multiple individual entities (e.g., board game),
    # as well as non-containment entities as well (e.g., City district, Campsite)
    # and non-tangible entities (e.g., Scenic viewpoint, Tree plantation).
    count = 0
    if not os.path.exists(prompt_folder):
        os.makedirs(prompt_folder)

    # Write prompts to get individual entities
    for response_filename in sorted(os.listdir(response_folder)):
        if response_filename.endswith('.txt'):
            with open(response_folder + response_filename, 'r') as f_read:
                line = f_read.readline()
                while not re.match("^[0-9]+\.", line):
                    line = f_read.readline()
                while re.match("^[0-9]+\.", line):
                    if 'count noun' in line.lower():
                        file_number = str(int(count / 50))
                        if len(str(int(count / 50))) == 1:
                            file_number = '0' + file_number
                        with open(prompt_folder + prompt_filename + file_number + '.txt', 'a') as f_write:
                            if count % 50 == 0:
                                f_write.write("Could you classify the following 50 nouns based on whether "
                                              "they are typically described as an entity on their own, or "
                                              "composed of multiple standalone entities? Add \" - a single entity\" "
                                              "\" - a group of components but commonly referred to as a single item\", "
                                              "or \" - a group of multiple standalone items\" "
                                              "accordingly after the nouns in the list.\n\n")
                            f_write.write(str(count % 50 + 1) + ". " +
                                          line.split(".", 1)[1].rsplit("- ", 1)[0].rsplit(" -", 1)[0].strip())
                            if count % 50 != 49:
                                f_write.write("\n")
                        count += 1
                    line = f_read.readline()

    reformat_last_file(prompt_folder, prompt_filename, file_number)


@retry(tries=4)
def receive_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,  #default
        top_p=1,  #default
        max_tokens=1024
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message


def write_response(prompt_folder, response_folder, response_filename):
    if not os.path.exists(response_folder):
        os.makedirs(response_folder)

    for prompt_filename in sorted(os.listdir(prompt_folder)):
        if prompt_filename.endswith('.txt'):
            file_ending = prompt_filename.rsplit("-", 1)[-1]
            with open(prompt_folder+prompt_filename, 'r') as f_read:
                with open(response_folder+response_filename+file_ending, 'w') as f_write:
                    prompt = f_read.read()
                    prompt_noun_list = prompt.split("\n\n")[-1].split("\n")
                    response = receive_response(prompt)
                    # make sure the names of the nouns are not altered
                    while any(prompt_noun not in response for prompt_noun in prompt_noun_list):
                        response = receive_response(prompt)
                    f_write.write(response)


def obtain_noun_list(response_folder, save_file, exclude_list, ending_list):
    result = set()
    for filename in os.listdir(response_folder):
        if filename.endswith('.txt'):
            with open(response_folder+filename, 'r') as f_read:
                line = f_read.readline()
                while not re.match("^[0-9]+\.", line):
                    line = f_read.readline()
                while re.match("^[0-9]+\.", line):
                    if 'standalone items' not in line.lower():
                        noun = line.split(".", 1)[1].rsplit("- ", 1)[0].strip()
                        if (noun not in exclude_list) and all(not noun.endswith(' '+end) for end in ending_list):
                            result.add(noun)
                    line = f_read.readline()

    noun_list = list(result)
    noun_list.sort()
    with open(save_file, 'w') as f_write:
        for noun in noun_list[:-1]:
            f_write.write(noun + "\n")
        f_write.write(noun_list[-1])


def main():
    # Obtain a csv file with Wikidata entities
    if len(sys.argv) < 2:
        raise ValueError("Please specify the Wikidata file. "
                         "Example: ``python3 src/freqnouns.py nouns/filtered_wikidata.csv``")
    elif len(sys.argv) > 2:
        raise ValueError('Too many arguments are given. Please specify the Wikidata file only.')

    openai.api_key = OPENAI_API_KEY
    # simple machines and bio/organism-related terms
    non_phys_nouns = ['Lever', 'Wheel and axle', 'Pulley', 'Inclined plane', 'Wedge', 'Screw',
                      'Animal product', 'Unidentified flying object', 'Flying saucer']
    general_nouns = ['Building', 'Collectable', 'Dwelling', 'Gadget', 'Gift', 'Human body',
                     'Motor vehicle', 'Used car', 'Used good']
    group_endings = ['system']

    print("Reading the Wikidata file...")
    # Read the Wikidata file
    wiki_file = sys.argv[1]
    wiki_df = pd.read_csv(wiki_file, header=0, sep='\t').fillna('')

    print(f'Started the filtering process. See ``{PROMPT_FOLDER}`` directory to check progress.')
    easy_folder = PROMPT_FOLDER + 'easy-nouns/'
    phys_folder = PROMPT_FOLDER + 'phys-nouns/'
    count_folder = PROMPT_FOLDER + 'count-nouns/'
    indiv_folder = PROMPT_FOLDER + 'indiv-nouns/'

    print("Processing well-perceived nouns...")
    write_easy_prompts(easy_folder+'prompt/', 'easy-prompt-', wiki_df)
    write_response(easy_folder+'prompt/', easy_folder+'response/', 'easy-nouns-')

    print("Processing physical object nouns...")
    write_phys_prompts(easy_folder+'response/', phys_folder+'prompt/', 'phys-prompt-')
    write_response(phys_folder+'prompt/', phys_folder+'response/', 'phys-nouns-')

    print("Processing count noun objects...")
    write_count_prompts(phys_folder + 'response/', count_folder + 'prompt/', 'count-prompt-')
    write_response(count_folder + 'prompt/', count_folder + 'response/', 'count-nouns-')

    print("Processing individual-entity nouns...")
    write_indiv_prompts(count_folder + 'response/', indiv_folder + 'prompt/', 'indiv-prompt-')
    write_response(indiv_folder + 'prompt/', indiv_folder + 'response/', 'indiv-nouns-')

    obtain_noun_list(indiv_folder+'response/', SAVE_FILE, non_phys_nouns+general_nouns, group_endings)
    print("Done!")


if __name__ == "__main__":
    main()