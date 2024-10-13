from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)
import time
import re
import pandas as pd

RPM = 10000  # rate limit per minute
DELAY = 60.0 / RPM  # calculate the delay based on your rate limit


def receive_response(message, max_tokens, model, client):
    if not isinstance(message, list):
        message = [{"role": "user", "content": message}]

    timeout = time.time() + 60 * 15  # time limit set for 15 minutes
    while True:
        try:
            response = retry_prompt(message, max_tokens, model, client)
            break
        except:
            pass

        if time.time() > timeout:
            print(f'Suspending because an error from OpenAI.\n{response}')
            break

    return response


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def retry_prompt(message_list, max_tokens, model, client):
    # Sleep for the delay
    time.sleep(DELAY)
    response = client.chat.completions.create(
        model=model,
        messages=message_list,
        temperature=1,  # default
        top_p=1,  # default
        max_tokens=max_tokens
    )
    response_message = response.choices[0].message.content
    return response_message


def confirm_subtype(subtype, supertype, max_tokens, renaming_max_tokens, model, client):
    if subtype == '-':
        return '-'

    question = f"""\
Choose the most accurate response from below.
1) "{subtype}" is an appropriate name for a type of {supertype}.
2) The ill-formed term "{subtype}" doesn't necessarily indicate it's {supertype}.
3) "{subtype}" is not an appropriate name, but "{subtype}" describes a type of {supertype}.
4) "{subtype}" does not belong to {supertype}."""
    answer_choice = ""
    while not answer_choice:
        response = receive_response(question, max_tokens, model, client)
        answer_choice = re.search(r"\d[\)|\.]", response)
    answer_choice = answer_choice.group(0)
    if answer_choice[0] == '1':
        return subtype
    elif answer_choice[0] in {'2', '3'}:
        return rename_subtype(subtype, supertype, renaming_max_tokens, model, client)
    else:
        return ''


def rename_subtype(subtype, supertype, max_tokens, model, client):
    question = f'If necessary, convert "{subtype}" to an appropriate and correct noun phrase that accurately refers to a type of {supertype}, distinguishing "{subtype}" from other types of {supertype}. Otherwise, you may just return "{subtype}".\nWrite the best answer with quotation marks.'

    response = receive_response(question, max_tokens, model, client)
    name_in_quotes = re.search(r'"(.*?)"', response)
    while not name_in_quotes:
        response = receive_response(question, max_tokens, model, client)
        name_in_quotes = re.search(r'"(.*?)"', response)

    # get the first mention within quotation marks
    return name_in_quotes.group(0).strip('"')


def verbose_materials(response):
    for comb in ['combination of', 'combinations of', 'a blend of', 'blends of', 'a mix of', 'mixes of']:
        if (comb in response) and\
                not (response.endswith(comb+' both') or any((comb+' both'+ending) in response for ending in {'.', '\n', ')'})):
            return True
    return False


def find_last_conjuction(text, conj_list):
    last_conj = ''
    last_conj_ind = 0
    for conj in conj_list:
        if text.rfind(conj) > last_conj_ind:
            last_conj_ind = text.rfind(conj)
            last_conj = conj
    return last_conj, last_conj_ind


def extract_material_str(material_str):
    material_keywords = {'alloy', 'aluminium', 'aluminum',
                        'bamboo', 'bone', 'brass', 'brick', 'bronze',
                        'cardboard', 'cement', 'cements', 'ceramic',
                        'chemical', 'clay', 'concrete', 'copper', 'cotton',
                        'electronic', 'fabric', 'felt', 'fiber', 'foam',
                        'gel', 'gem', 'glass', 'gold', 'hardboard', 'hardwood',
                        'iron', 'ivory', 'ivories', 'leather', 'linen', 'liquid', 'lumber',
                        'metal', 'nickel', 'nylon', 'paper', 'plastic', 'platinum', 'plywood',
                        'polyester', 'porcelain', 'powder', 'quartz', 'rubber', 'satin', 'silicon', 'silk',
                        'silver', 'slate', 'spandex', 'steel', 'stone', 'synthetic',
                        'textile', 'tile', 'titanium', 'vinyl', 'wood', 'wool'}

    # if materials are listed after 'various/varied', then take the listed materials
    # otherwise, simplify it by changing it to 'various materials'
    if any(material_str.lower().startswith(varying_token) for varying_token in {'varied', 'varies', 'various', 'vary'}) or \
            (' varies' in material_str.lower() or ' vary' in material_str.lower()):
        has_include_token = re.search(r'^(includ)| includ|\(includ', material_str.lower())
        if has_include_token:
            material_str = material_str[has_include_token.end():].split(" ", 1)[-1]
        elif any(any(token.lower().startswith(keyword) for keyword in material_keywords) for token in
                 re.split(r'\s|\(', material_str)):
            for token in re.split(r'\s|\(', material_str):
                if any(token.lower().startswith(keyword) for keyword in material_keywords):
                    material_str = material_str[material_str.find(token):].split(", depending ")[0].split(" depending ")[0]
                    break
        else:
            material_str = 'various materials'

    # remove parts whose material doesn't exist; e.g., "N/A", "no materials", "none".
    elif (not material_str) or material_str in {'—', '-'} or \
            any(material_str.lower().startswith(neg_token) for neg_token in {'not ', 'no ', 'none', 'n/a'}) or \
            (('no ' in material_str.lower() or 'not ' in material_str.lower()) and\
             ('material' in material_str.lower() or 'physical' in material_str.lower())):
        return '-'

    # convert a sentence to a phrase with a list of nouns
    found_predicate = re.search(r'(be|is|are) ([\sa-zA-Z]*)made [a-zA-Z]* ', material_str)
    if found_predicate:
        material_str = material_str[found_predicate.end():]

    # clean up
    material_str = re.sub(r'(\s)*\([^)]*\)', '', material_str)
    material_str = material_str.split(", etc.", 1)[0].split(")", 1)[0].strip('. ')

    # convert 'combination/mix/blend of both' to 'and/or'
    conj_list = [' and/or ', ' or ', ' and ', ', ']
    comb_list = list()
    for comb_head, plural in {'combination': 's', 'blend': 's', 'mix': 'es'}.items():
        for comb_word in ['a ' + comb_head, comb_head + plural]:
            for comb_tail in [' of both', ' thereof']:
                # e.g., 'A, B, a blend thereof'
                if comb_word + comb_tail in material_str:
                    comb_list.append(comb_word + comb_tail)
            # e.g., 'A, B, or a blend'
            if any(any(c1 + comb_word + c2 in material_str or material_str.endswith(c1 + comb_word)
                       for c1 in conj_list) for c2 in conj_list):
                comb_list.append(comb_word)
    if any(any(c1 + 'both' + c2 in material_str or material_str.endswith(c1 + 'both')
               for c1 in conj_list) for c2 in conj_list):
        comb_list.append('both')

    for comb_phrase in comb_list:
        two_materials, rest_materials = material_str.split(comb_phrase, 1)
        _, conj_ind = find_last_conjuction(two_materials, conj_list)
        two_materials = two_materials[:conj_ind]
        two_materials = two_materials.strip(',')
        last_conj, last_conj_ind = find_last_conjuction(two_materials, conj_list)
        material_str = two_materials[:last_conj_ind] + ' and/or ' + two_materials[last_conj_ind + len(
            last_conj):] + rest_materials

    return material_str


def append_rows(df, col_names, col_values):
    if len(col_names) != len(col_names):
        raise ValueError("The two lists `col_names` and `col_values` must have the same length.")

    row_dict = dict()
    for i in range(len(col_names)):
        row_dict[col_names[i]] = col_values[i]

    new_df = pd.DataFrame([row_dict])
    return pd.concat([df, new_df], axis=0, ignore_index=True)
