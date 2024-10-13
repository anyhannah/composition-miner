import copy
import inflect
import io
import os
from openai import OpenAI
import pandas as pd
from pathlib import Path
import re
from nltk.stem import WordNetLemmatizer
from prompt import *

INFLECT_ENGINE = inflect.engine()
WNL = WordNetLemmatizer()

if Path('nouns/nounlist.txt').exists():
    # custom noun list
    NOUN_FILE = 'nouns/nounlist.txt'
else:
    # provided noun list
    NOUN_FILE = 'data/nounlist.txt'
RESULT_FOLDER = 'result/'
if not Path(RESULT_FOLDER).exists():
    os.makedirs(RESULT_FOLDER)

LOG_FILE = RESULT_FOLDER+'log/zero_log.txt'
RESULT_FILE = RESULT_FOLDER+'csv/zero_interm_result.csv'
CLEAN_RESULT_FILE = RESULT_FOLDER+'csv/zero_result.csv'
MAX_TOKENS = 1024
MODEL = "gpt-4-1106-preview"
OPENAI_API_KEY = '[API_KEY]'
CLIENT = OpenAI(api_key=OPENAI_API_KEY)


def find_plural(word):
    # check whether the word is plural or singular
    # by lemmatizing the last token of the word
    if " (" in word:
        word = word.split(" (", 1)[0]

    # words including prepositions are ignored
    if " in " in word or " of " in word:
        return word

    tokens = word.strip().split(" ")
    last_word = tokens[-1]
    lemma = WNL.lemmatize(last_word, 'n')
    if last_word != lemma:  # already plural
        return word

    plural = INFLECT_ENGINE.plural(last_word)
    if last_word != word:
        # word consists of more than one token
        return " ".join(tokens[:-1]) + " " + plural
    else:
        return plural


def find_article(item, plural_item):
    # Add an article when the item is an appropriate singular type
    if item == plural_item:
        return ""

    if any(item.startswith(vowel_exception) for vowel_exception in {'europe', 'one', 'ufo', 'uni', 'use', 'utensil', 'u-'}):
        return "a "
    elif any(item.startswith(vowel) for vowel in ({'a', 'e', 'i', 'o', 'u'}|{'heir', 'herb', 'honor', 'hour'})):
        return "an "
    else:
        return "a "


def extract_likely_items(response):
    # We allow "likely", "probably likely", while "probably unlikely" and "unlikely" items are excluded
    result = list()
    response_io = io.StringIO(response)
    line = response_io.readline()
    while line and not ((re.match(r"^[0-9]+\.", line) or re.match(r"^[0-9]+\)", line) or line.startswith('- '))):
        line = response_io.readline()
        while (re.match(r"^[0-9]+\.", line) or re.match(r"^[0-9]+\)", line) or line.startswith('- ')):
            if 'unlikely' not in line.lower():
                if re.match(r"^[0-9]+\.", line):
                    item = line.split(".", 1)[1].strip()
                elif re.match(r"^[0-9]+\)", line):
                    item = line.split(")", 1)[1].strip()
                else:
                    item = line.split("-", 1)[1].strip()
                item = item.split(" - ")[0].split(" – ")[0].split(": ")[0].split(" (")[0]
                result.append(item.strip(". "))
            line = response_io.readline()
    response_io.close()
    return result


def prompt_response(message, max_tokens, model, client=CLIENT):
    return receive_response(message, max_tokens, model, client)


def q_has_types(item, max_tokens=MAX_TOKENS, model=MODEL):
    question = f'Are there any essential, non-optional parts\n1) that are present in one type of {item} but absent in another and\n2) that would be recognized by most people?\nSimply say "yes" or "no".'

    # ask for three times and return the most frequent answer
    res = {'yes': 0, 'no': 0}
    for i in range(3):
        response = prompt_response(question, max_tokens, model)
        if 'yes' not in response.lower():
            res['no'] += 1
        else:
            res['yes'] += 1
        # same answers in the first two times
        if i == 1 and min(res.values()) == 0:
            return max(res, key=res.get)

    return max(res, key=res.get)


def q_list_types(quoted_item, max_tokens=MAX_TOKENS, model=MODEL):
    question = f"In numbered points, please simply list physically distinct types of {quoted_item}, where each type is distinguished by unique, externally visible, essential parts.\n\nExclude any categories that share the same essential external components and functions. The listed categories should reflect differences in their primary operation rather than just external design variations or connections.\n\nAlso, avoid from your list any categories that merely represent design variations, subtypes, or alternate names for the same tool. Format each entry as a complete noun without using 'traditional', 'and', 'or', nouns indicating materials, or any prepositional phrases such as 'with' in the names."
    response = prompt_response(question, max_tokens, model)
    response = "\n".join([re.sub(r' \([^)]*\)', '', line.strip(".")) for line in response.split("\n")
                          if (line.strip() and line.strip()[0].isdigit())])
    return response


def q_likely_types(item, type_list, max_tokens=MAX_TOKENS, model=MODEL):
    question = f'How likely would the following types of {item} be recognized by most people? Add " - [likely / probably likely / probably unlikely / unlikely] recognized by most people" after the nouns in the list. Please do not alter the names within parentheses.\n\n{type_list}'
    return prompt_response(question, max_tokens, model)


def q_distinct_parts(item, max_tokens=MAX_TOKENS, model=MODEL):
    question = f'How many parts does {item} have? Specifically, how many clearly distinct parts that are attached to it or inseparable from it? Please simply say the number of parts.'
    return prompt_response(question, max_tokens, model)


def q_item_materials(item, max_tokens=MAX_TOKENS, model=MODEL):
    question = f'In one line, please list solely the types of materials that {item} are typically made of. Avoid using "sometimes", and connect the materials with a conjunction, e.g., \'glass, plastic, and/or metal\'. Exclude any materials used for joining, stitching or dying.\n\nHere are the conjunctions you can use:\n- "and": all listed materials are typically used together\n- "or": each of the materials from the list is used exclusively\n- "and/or": some of the listed materials are typically used in combination.'
    response = prompt_response(question, max_tokens, model)
    return response


def q_same_materials(item, max_tokens=MAX_TOKENS, model=MODEL):
    question = f'Are distinct parts of {item} made of the same materials? Say "yes" or "no".'
    return prompt_response(question, max_tokens, model)


def q_parts_materials(article_item, max_tokens=MAX_TOKENS, model=MODEL):
    article, item = article_item
    question = f'1) Starting your paragraph with "<Parts>\\n", in numbered points, please list clearly distinct, essential parts of {article}{item} with succinct descriptions followed by ":". For each part, insert a new line that starts with "- Optional:". Answer with "yes" or "no".\n\n2) Starting your paragraph with "<Materials>: ", in new bullet points, please list solely the materials that {article}typical {item} is entirely made of. Avoid using "sometimes", and connect the materials with a conjunction, e.g., \'<Materials>: glass, plastic, and/or metal\'. Exclude any materials used for joining, stitching or dying. Here are the conjunctions you can use.\n- "and": all listed materials are typically used together\n- "or": each of the materials from the list is used exclusively\n- "and/or": some of the listed materials are typically used in combination.\n\nKeep your answers very simple, in terms a second-grader would understand.'
    return ask_multiple_times(question, max_tokens, model)


def q_different_parts_materials(item, max_tokens=MAX_TOKENS, model=MODEL):
    question = f'In numbered points, please list the clearly distinct, essential parts of {item} that are attached to it or inseparable from it, with succinct descriptions following ":". Things that have multiple independent uses, such as \'battery\', don\'t count as a part. You may use "internal mechanism" as a part for anything that is not visible from the outside.\n\nFor each part, insert a new line that starts with "- Optional:". Answer with "yes" or "no".\n\nThen again, for each part, insert a new line that starts with "- Materials:" and mention the materials the part is typically made of. List the materials, avoiding using "sometimes", and connect the materials with a conjunction, e.g., \'- Materials: glass, plastic, and/or metal\'. Here are the conjunctions you can use.\n- "and": all listed materials are typically used together\n- "or": each of the materials from the list is used exclusively\n- "and/or": some of the listed materials are typically used in combination.\n\nKeep your answers very simple, in terms a second-grader would understand.'
    return ask_multiple_times(question, max_tokens, model)


def get_problem_prompt(response):
    if re.search(r"(-|[0-9]\.) (additional|Additional|various|Various) (.*)[^\n]:", response):
        problem_prompt = """Try to avoid using "additional" and "various" in parts' names."""
        return problem_prompt

    detect_mechanism = re.findall(r'(-|[0-9]\.) (?![i|I]nternal)(.*) (mechanism|system):', response)
    if detect_mechanism:
        mechanism_part = ' '.join(detect_mechanism[0][1:])
        problem_prompt = """The part "{sample}" doesn't seem to refer to a tangible part."""
        return problem_prompt.replace("{sample}", mechanism_part)

    return ''


def ask_multiple_times(question, max_tokens, model):
    response = prompt_response(question, max_tokens, model)

    # to avoid 'additional', 'various', 'mechanism', 'system' in parts' names
    problem_prompt = get_problem_prompt(response)
    while problem_prompt:
        message_list = [{"role": "user", "content": question},
                        {"role": "assistant", "content": response},
                        {"role": "user", "content":
                            problem_prompt + "\n\n" + question.split("materials with a conjunction", 1)[0]
                            + "materials with a conjunction."}]
        response = prompt_response(message_list, max_tokens, model)
        problem_prompt = get_problem_prompt(response)

    return response


def extract_subtypes(supertype, max_tokens=MAX_TOKENS, model=MODEL, client=CLIENT):
    noun_dict = dict()
    while len(noun_dict.keys()) == 0:
        # find subtypes
        a_list_types = q_list_types(f'"{supertype}"')
        a_likely_types = q_likely_types(supertype.lower(), a_list_types)
        subtype_list = extract_likely_items("\n" + a_likely_types)
        a_likely_types += "\n"
        renamed_trace = ""
        excluded_trace = ""

        for subitem in subtype_list:
            # rename subtypes
            new_subtype = confirm_subtype(subitem, supertype, 512, 128, model, client)
            if not new_subtype:
                excluded_trace += f"{subitem} is removed from the subtypes of {supertype}.\n"
                continue
            if new_subtype.lower() in [lower_type.lower() for lower_type in noun_dict.keys()]:
                # If the new name for the subtype already exists as another subtype, find subtypes again.
                noun_dict = dict()
                break
            if new_subtype.lower() == re.sub(r' \([^)]*\)$', '', supertype.lower()):
                # If the original subtype name is the same as the supertype name, remove it.
                excluded_trace += f"{subitem} is removed from the subtypes of {supertype}.\n"
                continue
            if new_subtype.lower() != subitem.lower():
                renamed_trace += f"{subitem} is renamed as {new_subtype}.\n"
            noun_dict[new_subtype] = list()

    for new_subtype in noun_dict.keys():
        # find subsubtypes
        a_has_types = q_has_types(new_subtype, 10, model)
        if a_has_types == 'no':
            noun_dict[new_subtype] = list()
        else:
            if supertype.endswith(')'):
                category = " (" + supertype.rsplit("(", 1)[1].lower()
            elif supertype.lower() not in new_subtype.lower():
                category = f" (a type of {supertype.lower()})"
            else:
                category = ""
            while len(noun_dict[new_subtype]) == 0:
                a_list_types = q_list_types(f'"{new_subtype}{category}"')
                a_likely_subsubtypes = q_likely_types(new_subtype.lower(), a_list_types)
                subsubtype_list = extract_likely_items("\n" + a_likely_subsubtypes)
                a_likely_types_temp = "Subtype: " + new_subtype + "\n" + a_likely_subsubtypes + "\n"
                renamed_trace_temp = ""
                excluded_trace_temp = ""

                # rename subsubtypes
                for subsubtype in subsubtype_list:
                    new_subsubtype = confirm_subtype(subsubtype, new_subtype, 512, 128, model, client)
                    if not new_subsubtype:
                        excluded_trace_temp += f"{subsubtype} is removed from the subtypes of {new_subtype}.\n"
                        continue
                    if new_subsubtype.lower() in [lower_type.lower() for lower_type in noun_dict[new_subtype]]:
                        # If the new name for the subsubtype already exists as another subsubtype, find subsubtypes again.
                        noun_dict[new_subtype] = list()
                        break
                    if new_subsubtype.lower() == re.sub(r' \([^)]*\)$', '', supertype.lower()) or\
                            new_subsubtype.lower() == new_subtype.lower() or\
                            new_subsubtype.lower() in [sub_name.lower() for sub_name in noun_dict.keys()]:
                        # If the subsubtype is the same as the supertype or subtype name, remove it.
                        # If the subsubtype exists as another subtype, remove it.
                        excluded_trace_temp += f"{subsubtype} is removed from the subtypes of {new_subtype}.\n"
                        continue
                    if new_subsubtype.lower() != subsubtype.lower():
                        renamed_trace_temp += f"{subsubtype} is renamed as {new_subsubtype}.\n"

                    noun_dict[new_subtype].append(new_subsubtype)

            a_likely_types += a_likely_types_temp
            renamed_trace += renamed_trace_temp
            excluded_trace += excluded_trace_temp

    return noun_dict, a_likely_types, renamed_trace, excluded_trace


def organize_subtypes(supertype, noun_dict):
    # If there exists one subtype and no subsubtypes, ignore the subtype
    # (treating it as the same type as the item itself)
    # If there is one subtype and one subsubtype, or there is no subtype/subsubtypes found, ignore them as well.
    if (len(noun_dict) == 1 and not any(noun_dict.values())) or \
            (len(noun_dict) == 1 and len(list(noun_dict.values())[0]) == 1) or \
            len(noun_dict) == 0:
        noun_dict = dict()
        noun_dict[supertype] = list()
    else:
        # If there is a subtype (not the only subtype) and only one subsubtype, then ignore the subsubtype.
        noun_dict_temp = copy.deepcopy(noun_dict)
        for subtype, subsub_list in noun_dict_temp.items():
            if len(subsub_list) == 1:
                noun_dict[subtype] = list()

    return noun_dict


def wrong_parts_materials_format(response):
    # The response for parts and materials should have multiple lines
    # starting with either "1" or "-".
    response = response.split("<Parts>\n", 1)[-1].split("<Parts>\\n", 1)[-1].strip()
    if response.count("\n") < 1:
        return True
    if any(any(paragraph.strip().startswith(bullet_point) for bullet_point in {"- ", "1. ", "1) "})
           for paragraph in response.split("\n\n")):
        return False
    first_line, rest_par = response.split("\n", 1)
    return not (("Here's " in first_line or len(first_line.split(" ")) < 10) and\
                any(rest_par.strip().startswith(bullet_point) for bullet_point in {"- ", "1. ", "1) "}))


def ensure_material_semantics(question_type, response, question_func, question_params):
    parts_optionality, _, material_response, dependent_material = extract_parts_materials(question_type, response)
    if materials_too_long(material_response) or parts_too_long(material_response) or\
            ('unknown' in material_response.lower()) or\
            verbose_materials(material_response.lower()) or dependent_material:
        # Prompt the model for the composition materials again
        response = question_func(question_params)
        _, _, material_response, dependent_material = extract_parts_materials(question_type, response)
        while verbose_materials(material_response.lower()) or dependent_material:
            response = question_func(question_params)
            _, _, material_response, dependent_material = extract_parts_materials(question_type, response)
        if materials_too_long(material_response) or parts_too_long(material_response) or\
                'unknown' in material_response.lower():
            response = question_func(question_params)
            _, _, material_response, dependent_material = extract_parts_materials(question_type, response)
            while verbose_materials(material_response.lower()) or dependent_material:
                response = question_func(question_params)
    return response


def materials_too_long(material_str):
    material_str = re.sub(r'\([^)]*\)( *)', '', material_str).strip()
    return any(len(m.split()) > 5 for m in re.split(r" and\/or | or | and |,|, ", material_str))


def parts_too_long(parts_optionality):
    if "\n" in parts_optionality:
        for line in parts_optionality.split("\n"):
            if re.sub(r' [\*]*\([^)]*\)[\*]*', '', line).split(": ", 1)[0].count(" ") > 10:
                return True
    return False


def extract_parts_materials(question_type, response):
    parts_optionality = ''  # in '{part}: {optionality}' format
    materials = ''  # in '{part}: {materials}' format
    material_response = ''  # material information as it is returned by the model
    material_str = ''  # simplified, refined material list
    dependent_material = False

    if question_type == 'A':
        parts_optionality = '-: F'
        material_response = response.strip(".")
        materials = extract_material_str(response.strip("."))

    elif ("<Parts>\n" in response or "<Parts>\\n" in response) and re.search(r'<"?Materials"?>', response, re.IGNORECASE):
        material_splitter = re.search(r'<"?Materials"?>', response, re.IGNORECASE)[0]
        parts, material_par = response.split("<Parts>\n", 1)[-1].split("<Parts>\\n", 1)[-1].split(material_splitter, 1)
        material_par = material_par.strip(': ')

        # materials
        if material_par.strip().count("\n") < 1:
            material_response = material_par.strip("-. \n")
            material_str = extract_material_str(material_response)
        else:
            # put together the materials that were listed over multiple lines into one line
            conjunctions = {' or '}
            for material in material_par.strip().split('\n'):
                material = material.strip('-. ')
                if 'and/or ' in material:
                    conjunctions.add(' and/or ')
                if material.startswith('and ') or ' and ' in material:
                    conjunctions.add(' and ')
                material = material.lstrip('and/or ').lstrip('or ').lstrip('and ')
                for indiv_material in re.split(r" or | and | and\/or ", material):
                    material_str += indiv_material + ", "
            material_str = material_str.rstrip(', ')
            if ', ' in material_str:
                conj = max(conjunctions, key=len)  # priority goes 'and/or' > 'and' > 'or'
                if material_str.count(', ') == 1:
                    material_str = conj.join(material_str.rstrip(', ').rsplit(", ", 1))
                else:
                    material_str = (',' + conj).join(material_str.rstrip(', ').rsplit(", ", 1))
            material_response = material_str

        # parts
        part_count = 1
        for part_response in re.split(r"[0-9]+\. |[0-9]+\) ", parts.strip()):
            part_response = part_response.strip()
            if part_response and ':' in part_response:
                try:
                    part, optionality = part_response.split("- Optional: ", 1)
                except:
                    part = part_response.split(":", 1)[0]
                    if '\n' in part_response and ':' in part_response.split("\n", 1)[1]:
                        optionality = part_response.split("\n", 1)[1].split(":", 1)[1].strip()
                    else:
                        optionality = 'no'

                part = re.split(r'\:\s*(?![^()]*\))', part.strip(), 1)[0]
                if part.startswith('"') and part.endswith('"'):
                    # If a part name is surrounded by quotation marks, remove them.
                    part = re.sub(r"(^\")|(\"$)", "", part)
                part = re.sub(r"\*", "", part)  # remove asterisks
                if '\n' in part:  # remove part description in a new line
                    part = part.split("\n", 1)[0]
                if ' (' in part:
                    part_no_paren = re.sub(r' \([^)]*\)', '', part)
                    if part_no_paren.lower() == 'internal mechanism':
                        # e.g., 'Internal mechanism (HVAC system)' becomes 'HVAC system'
                        part = re.search(r' \([^)]*\)', part).group(0)[2:-1]
                    elif len(re.findall(f'{part_no_paren} \\([a-zA-Z\\s]+\\):', parts)) > 1:
                        # e.g., Prong (middle), Prong (left), Prong (right)
                        # becomes Middle prong, Left prong, Right prong
                        if re.search(r' [A-Z]', part_no_paren):
                            part = re.search(r' \([^)]*\)', part).group(0)[2:-1] + ' ' + part_no_paren
                            part = part[0].upper() + part[1:]
                        else:
                            part = re.search(r' \([^)]*\)', part).group(0)[2:-1] + ' ' + part_no_paren[
                                0].lower() + part_no_paren[1:]
                            part = part[0].upper() + part[1:]
                    else:
                        # remove parentheses
                        part = part_no_paren
                part = part.strip('. ')
                if optionality.lower().startswith('no'):
                    optionality = 'F'
                else:
                    # including 'sometimes', 'varies', etc.
                    optionality = 'T'
                parts_optionality += f'{part}: {optionality}\n'
                materials += f'{part_count}. {part}: {material_str}\n'
                part_count += 1

    elif any(any(paragraph.startswith(bullet) for bullet in {"- ", "1. ", "1) "}) for paragraph in response.split("\n\n")):
        response = re.sub(r'\n+', '\n', response)
        part_count = 1
        while re.search(r'[-|—][\s]+material[s]*:', response, re.IGNORECASE):
            # encompasses the case variations of '- Materials:', '- Material:', '-  Materials:', '-\nMaterials:',
            # but any other variations, such as, '- Blade Materials' and '- (Materials:',
            # should have the prompt re-run in the earlier stage
            material_splitter = re.search(r'[-|—][\s]+material[s]*:', response, re.IGNORECASE)[0]
            first_par_ind = response.split(material_splitter)[0].count("\n") + 1 + material_splitter.count("\n")
            paragraph = "\n".join(response.split("\n")[:first_par_ind])
            response = "\n".join(response.split("\n")[first_par_ind:])
            material_str = paragraph.split(material_splitter)[1].split("\n", 1)[0].strip(". ")
            material_response += material_str + "\n"

            if paragraph.count("\n") < 2:
                # erroneous lines; e.g., materials given without a part name
                continue

            # the material is the same as another part's.
            search_twin_part = re.search(r'(same [a-zA-Z\s]+ as )|(same as )', material_str, re.IGNORECASE)
            if search_twin_part:
                new_material_str = ''
                new_material_conj = 'or'
                mentioned_material_str = material_str[:search_twin_part.start()]
                if any(mentioned_material_str.endswith(conj) or mentioned_material_str.endswith(conj + 'the ')
                       for conj in {' and/or ', ' and ', ' or '}):
                    new_material_conj_ind = max(mentioned_material_str.rfind(' and/or ', 1),
                                                mentioned_material_str.rfind(' and ', 1),
                                                mentioned_material_str.rfind(' or ', 1))
                    new_material_str = mentioned_material_str[:new_material_conj_ind].strip(',')
                    new_material_conj = mentioned_material_str[new_material_conj_ind:].strip().split()[0]
                twin_part = material_str[search_twin_part.end():]
                if twin_part.startswith('the '):
                    twin_part = twin_part[4:]
                first_token = twin_part.split()[0]
                twin_tokens = {twin_part, twin_part + 's', twin_part + 'es',
                               first_token, first_token.rstrip("'s"), " ".join(twin_part.split()[:2])}
                if len(twin_part.split()) > 1:
                    twin_tokens.add(twin_part.split()[1])
                for twin_token in twin_tokens:
                    twin_candidates = [m_line.split(": ", 1)[-1] for m_line in materials.split("\n")
                                       if ' ' + twin_token.strip(",.);-").lower() + ":" in m_line.lower()]
                    if len(twin_candidates) == 1:
                        # there can be only one matching part;
                        # e.g., top sheet vs. bottom sheet should be distinguished
                        if new_material_str:
                            # a matching part already exists
                            # e.g, the same material as the 'skirt' or 'sleeves'
                            temp_material_str = ', '.join([token.strip() for token in
                                                           re.split(r'[,]*( and\/or | and | or |,)',
                                                                    twin_candidates[0])
                                                           if (token.strip() not in new_material_str) and \
                                                           (token.strip() not in {',', 'and/or', 'and', 'or'})])
                            if temp_material_str:
                                new_material_str += ', ' + (', '+new_material_conj+' ').join(temp_material_str.rsplit(', ', 1))
                        else:
                            new_material_str = twin_candidates[0]
                material_str = new_material_str
                if re.search(r'(same [a-zA-Z\s]+ as )|(same as )', material_str, re.IGNORECASE):
                    dependent_material = True
            else:
                if 'same ' in material_str.lower() and\
                        any(delimiter in material_str for delimiter in {'(', '-', '–', ':', ';'}):
                    material_str = re.split(r'\(|-|–|:|;',
                                            re.split('same ', material_str, re.IGNORECASE)[1])[-1].strip("). ")

            material_str = extract_material_str(material_str)
            if material_str == '-':
                # no physical materials exist
                continue

            # each "paragraph" consists of three or four lines:
            # part, (description), optionality, and material information
            if not ('- Optional:' in paragraph.split("\n")[1] or '- Description:' in paragraph.split("\n")[1]) and \
                    not (paragraph.strip()[0].isdigit() or paragraph.strip()[0] == '-') and \
                    paragraph.split("\n")[1].strip()[0].isdigit():
                # first line has unnecessary explanations
                paragraph = "\n".join(paragraph.split("\n")[1:])
            if '- Optional' not in paragraph:
                continue
            if paragraph[0] == '-' or paragraph[0].isdigit():
                paragraph = paragraph.split(" ", 1)[-1]
            part = re.split(r"(^Part: )|(^Part name: )", paragraph.split("\n", 1)[0])[-1]
            part = re.split(r'\:\s*(?![^()]*\))', part, 1)[0].strip()  # split by ":" except when it's within parentheses
            if part.startswith('"') and part.endswith('"'):
                # remove quotation marks, asterisks, parentheses from the part name.
                part = re.sub(r"(^\")|(\"$)", "", part)
            part = re.sub(r"\*", "", part)
            if ' (' in part:
                part_no_paren = re.sub(r' \([^)]*\)', '', part)
                if part_no_paren.lower() == 'internal mechanism':
                    # e.g., 'Internal mechanism (HVAC system)' becomes 'HVAC system'
                    part = re.search(r' \([^)]*\)', part).group(0)[2:-1]
                elif len(re.findall(f'{part_no_paren} \\([a-zA-Z\\s]+\\):', paragraph + response)) > 1:
                    # e.g., Prong (middle), Prong (left), Prong (right)
                    # becomes Middle prong, Left prong, Right prong
                    if re.search(r' [A-Z]', part_no_paren):
                        part = re.search(r' \([^)]*\)', part).group(0)[2:-1] + ' ' + part_no_paren
                        part = part[0].upper() + part[1:]
                    else:
                        part = re.search(r' \([^)]*\)', part).group(0)[2:-1] + ' ' + part_no_paren[0].lower() + part_no_paren[1:]
                        part = part[0].upper() + part[1:]
                else:
                    # remove parentheses
                    part = part_no_paren
            part = part.strip('. ')

            optionality = paragraph.split("- Optional", 1)[1].split("\n", 1)[0].strip(": ")
            if optionality.lower().startswith('no'):
                optionality = 'F'
            else:
                optionality = 'T'

            parts_optionality += f'{part}: {optionality}\n'
            materials += f'{part_count}. {part}: {material_str}\n'
            part_count += 1

    else:
        # OpenAI did not provide the requested information
        parts_optionality = '-'
        materials = '-'
        material_response = response

    if materials.startswith("1.") and materials.strip().count("\n") == 0:
        # parts reduced to no distinct parts after removing non-material responses
        parts_optionality = '-: F'
        materials = materials.split(": ", 1)[-1]

    if not materials.strip() or materials == '-':
        parts_optionality, materials = '-', '-'

    # remove descriptions followed by a part name
    new_parts_optionality = parts_optionality.strip()
    new_materials = materials.strip()
    for dash in [' - ', ' – ', ' (']:
        if dash in parts_optionality:
            for line in parts_optionality.split('\n'):
                if dash not in line:
                    pass
                elif parts_optionality.count(line.split(dash, 1)[0] + dash) > 1:
                    # take the name after '-'; e.g., "internal mechanism - Pump", "internal mechanism - Heater"
                    new_line = line.split(dash, 1)[1].replace("):", ":")
                    new_parts_optionality = new_parts_optionality.replace(line, new_line)
                    new_materials = new_materials.replace('. ' + line.rsplit(': ', 1)[0] + ': ',
                                                          '. ' + new_line.rsplit(": ", 1)[0] + ": ")
                else:
                    # take the name before '-'
                    new_line = line.split(dash, 1)[0].rstrip(')') + ': '
                    new_parts_optionality = new_parts_optionality.replace(line, new_line + line.rsplit(': ', 1)[1])
                    new_materials = new_materials.replace('. ' + line.rsplit(': ', 1)[0] + ': ',
                                                          '. ' + new_line)

    return new_parts_optionality, new_materials, material_response.strip(), dependent_material


def append_rows(df, col_names, col_values):
    if len(col_names) != len(col_names):
        raise ValueError("The two lists `col_names` and `col_values` must have the same length.")

    row_dict = dict()
    for i in range(len(col_names)):
        row_dict[col_names[i]] = col_values[i]

    new_df = pd.DataFrame([row_dict])
    df = pd.concat([df, new_df], axis=0, ignore_index=True)
    return df


def clean_response_format(raw_df):
    col_names = ['item', 'subtype', 'subsubtype', 'optionality', 'response']
    clean_df = pd.DataFrame(columns=col_names)

    for ind, row in raw_df.iterrows():
        question_type = row['question']
        response = row['response']
        parts_optionality, materials, _, _ = extract_parts_materials(question_type, response)
        # change 'electronics' material to 'electronic materials'
        new_materials = re.sub(r':([^\n]+)electronics', ':\\1electronic materials', materials)
        clean_df = append_rows(clean_df, col_names,
                               [row['item'], row['subtype'], row['subsubtype'], parts_optionality, new_materials])
    return clean_df


def main():
    with open(NOUN_FILE, 'r') as f_read:
        col_names = ['item', 'subtype', 'subsubtype', 'question', 'response']
        result_df = pd.DataFrame(columns=col_names)
        supertype = f_read.readline().strip()
        while supertype:
            # Question 1 -- Has subtypes
            a_has_types = q_has_types(supertype, 10)

            # leave a quick progress log
            f_write = open(LOG_FILE, 'a')
            f_write.write('Noun: ' + supertype + "\n")

            if a_has_types == 'no':
                noun_dict = dict()
                noun_dict[supertype] = list()
            else:
                # Question 2 & 3 -- Find subtypes & subsubtypes
                noun_dict, a_likely_types, renamed_type_trace, excluded_type_trace = extract_subtypes(supertype)
                f_write.write(a_likely_types)
                if renamed_type_trace:
                    f_write.write("Renamed:\n" + renamed_type_trace)
                if excluded_type_trace:
                    f_write.write("Deleted:\n" + excluded_type_trace)

            f_write.write("\n")
            f_write.close()

            if supertype not in noun_dict.keys():
                noun_dict = organize_subtypes(supertype, noun_dict)

            for noun, subsub_list in noun_dict.items():
                # item_list becomes either only supertype, or only subitem(s), or only subsubitem(s).
                if subsub_list:
                    item_list = subsub_list
                else:
                    item_list = [noun]

                for cur_item in item_list:
                    subsubitem = ''
                    category = ''
                    supertype_tokens = supertype.lower().split()
                    if cur_item == supertype:
                        # Target: item
                        subitem = '-'
                        subsubitem = '-'
                    else:
                        subitem = noun

                    if not subsubitem:
                        if cur_item == noun:
                            # Target: subtype
                            subsubitem = '-'
                            if all(token not in noun.lower() for token in supertype_tokens) and\
                                    all(token.lower() not in supertype.lower() for token in noun.split()):
                                if " (" in supertype:
                                    category = f" ({supertype.rsplit(' (', 1)[1].lower()}"
                                else:
                                    category = f" (a type of {supertype.lower()})"
                        else:
                            # Target: subsubtype
                            subsubitem = cur_item
                            if all(token not in subsubitem.lower() for token in supertype_tokens) and\
                                    all(token.lower() not in subsubitem.lower() for token in noun.split()) and\
                                    all(token.lower() not in supertype.lower() for token in subsubitem.split()) and\
                                    all(token.lower() not in noun.lower() for token in subsubitem.split()):
                                if " (" in supertype:
                                    category = f" ({supertype.rsplit(' (', 1)[1].lower()}"
                                else:
                                    category = f" (a type of {re.sub(r' \([^)]*\)', '', noun.lower())})"


                    item_plural = find_plural(cur_item.lower())
                    article = find_article(cur_item.lower(), item_plural)

                    # Question 4 -- Has distinct parts
                    a_distinct_parts = q_distinct_parts(article + cur_item.lower() + category, 10)
                    if a_distinct_parts in {'0', '1'}:
                        # Question 5 -- Find materials for the type
                        a_item_materials = q_item_materials(item_plural + category, 256)
                        a_item_materials = ensure_material_semantics('A', a_item_materials, q_item_materials,
                                                                     item_plural + category)
                        result_df = append_rows(result_df, col_names,
                                                [supertype, subitem, subsubitem, 'A', a_item_materials])
                    else:
                        # Question 6 -- Different parts have uniform materials
                        a_same_materials = q_same_materials(article + cur_item.lower() + category, 10)
                        if "yes" in a_same_materials.lower():
                            # Question 7 -- Find parts and materials for the type
                            a_parts_materials = q_parts_materials((article, cur_item.lower() + category))
                            a_parts_materials = ensure_material_semantics('B', a_parts_materials, q_parts_materials, (article, cur_item.lower() + category))
                            if wrong_parts_materials_format(a_parts_materials):
                                a_parts_materials = q_parts_materials((article, cur_item.lower() + category))
                            result_df = append_rows(result_df, col_names,
                                                    [supertype, subitem, subsubitem, 'B', a_parts_materials])
                        else:
                            # Question 8 -- Find parts and materials for each part
                            a_diff_parts_materials = q_different_parts_materials(article + cur_item.lower() + category)
                            a_diff_parts_materials = ensure_material_semantics('C', a_diff_parts_materials, q_different_parts_materials, article + cur_item.lower() + category)
                            if not re.search(r'-[\s]+material[s]*:', a_diff_parts_materials, re.IGNORECASE) or\
                                    wrong_parts_materials_format(a_diff_parts_materials):
                                a_diff_parts_materials = q_different_parts_materials(article + cur_item.lower() + category)
                            result_df = append_rows(result_df, col_names,
                                                    [supertype, subitem, subsubitem, 'C', a_diff_parts_materials])


            result_df.to_csv(RESULT_FILE, sep='\t', index=False, encoding='utf-8')
            supertype = f_read.readline().strip()

        clean_df = clean_response_format(result_df)
        clean_df.to_csv(CLEAN_RESULT_FILE, sep='\t', index=False, encoding='utf-8')


if __name__ == "__main__":
    main()
