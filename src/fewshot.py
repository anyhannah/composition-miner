import os
from openai import OpenAI
import pandas as pd
from pathlib import Path
import re
import copy
from prompt import *

if Path('nouns/nounlist.txt').exists():
    # custom noun list
    NOUN_FILE = 'nouns/nounlist.txt'
else:
    # provided noun list
    NOUN_FILE = 'data/nounlist.txt'
RESULT_FOLDER = 'result/'
if not Path(RESULT_FOLDER).exists():
    os.makedirs(RESULT_FOLDER)

LOG_FILE = RESULT_FOLDER+'log/few_log.txt'
RESULT_FILE = RESULT_FOLDER+'csv/few_interm_result.csv'
CLEAN_RESULT_FILE = RESULT_FOLDER+'csv/few_result.csv'
MAX_TOKENS = 1024
MODEL = "gpt-4-1106-preview"
OPENAI_API_KEY = '[API_KEY]'
CLIENT = OpenAI(api_key=OPENAI_API_KEY)

PART_PROMPT = """Please list common categories and their sub-categories, and their constituent parts of the given entity. Each type must be distinguished solely by the unique presence of their essential parts or components. Only list essential parts, not in their variations in shape, size, material, or function. Please do not count chemical substances such as electrolyte as essential parts.

Alternatively, you may state "No distinct subtypes based on the constituent parts" instead of listing subtypes if there are no variations in the essential, unique parts that distinguish the subtypes. Then, indicate "Physical parts" underneath.

Please do not state any descriptive terms or clarifications within parentheses. Only indicate "(optional)" where applicable. You may use "internal mechanism" as a part for any components not visible externally but essential for function.

##
Entity 1: Barn
Subtypes 1:
1. English barn: walls, roof, floor, frame, three bays
2. Livestock barn: walls, roof, floor, frame, tack room, feed room (optional), drive bay, silo, stalls
3. Dairy barn: walls, roof, floor, frame, tack room, feed room, drive bay, silo, stalls, milk house, grain bin, indoor corral (optional)
4. Crop storage barn: walls, roof, frame, drive bay
5. Crib barn: walls, roof, cribs, roof shingles
6. Bank barn
6.a) New England barn: walls, roof, roof shingles, floor, tack room (optional), frame
6.b) Pennsylvania barn: walls, roof, roof shingles, floor, forbear, frame, gables (optional)
##
Entity 2: Saucer
Subtypes 2: No distinct subtypes based on the constituent parts.
Physical parts: No distinct parts
##
Entity 3: Paintbrush
Subtypes 3: No distinct subtypes based on the constituent parts.
Physical parts: handle, bristle, ferrule
##
Entity 4: Frying pan
Subtypes 4:
1. Stovetop frying pan: body, handle
2. Electric frying pan: body, handle, legs, lid (optional), lid knob (optional), power cord, thermostat
##
Entity 5: Glove (ice hockey)
Subtypes 5:
1. Skater's gloves: palm, back, fingers, padding
2. Blocker: palm, back, fingers, padding, forearm pad
3. Trapper: palm, back, fingers, padding, cuff, pocket, inner glove
##
Entity 6: {noun}
Subtypes 6:"""

SECOND_PART_PROMPT = """Please list common categories and their sub-categories, and their constituent parts of the given entity. If there are no variations in the essential, unique parts that distinguish the subtypes, you may state "No distinct subtypes based on the constituent parts" instead of listing subtypes.

##
Entity 6: {noun}
Subtypes 6:"""

MATERIAL_PROMPT = """Please list the materials that the listed parts of the given entity are typically made of. Exclude any materials used for joining, stitching or dying.

Allow any necessary repetition in materials across different parts. Avoid using "sometimes", "such as", and parentheses in your response. Connect the materials with one of the following conjunctions:
- "and": all listed materials are typically used together
- "or": each of the materials from the list is used exclusively
- "and/or": some of the listed materials are typically used in combination

##
Entity 1: Peripheral webcam (Webcam)
Parts: case, camera lens, image sensor, mount, interface 
Materials:
1. case: plastic
2. camera lens: plastic or glass
3. image sensor: electronics
4. mount: metal
5. interface: electronics, metal, and plastics
##
Entity 2: Paper cup
Parts: cup, cardboard lining, lid
Materials:
1. cup: paper
2. cardboard lining: plastic or wax
3. lid: plastic
##
Entity 3: Facial tissue
Parts: -
Materials: absorbent paper
##
Entity 4: Paintbrush
Parts: bristle, ferrule, handle
Materials:
1. bristle: animal hair, nylon, and/or polyester
2. ferrule: metal
3. handle: wood or plastic
##
Entity 5: Skater's gloves (ice hockey)
Parts: palm, back, fingers, padding
Materials:
1. palm: leather
2. back: leather and/or kevlar
3. fingers: leather and/or kevlar
4. padding: foam
##
Entity 6: {noun}
Parts: {parts}
Materials:"""


def prompt_response(message, max_tokens, model, client=CLIENT):
    return receive_response(message, max_tokens, model, client)


def ask_type_parts(noun, max_tokens=MAX_TOKENS, model=MODEL,
                   prompt=PART_PROMPT, second_prompt=SECOND_PART_PROMPT, client=CLIENT):
    part_dict = dict()
    while len(part_dict) == 0:
        part_response = ensure_part_format(noun, max_tokens, model)
        if 'No distinct subtypes' in part_response:
            # Ask about subtypes and parts again to ensure the answer
            part_response = ensure_part_format(noun, max_tokens, model)
        part_response = ensure_part_semantics(noun, part_response, prompt, second_prompt, max_tokens, model)
        part_dict, renamed_type_trace, excluded_type_trace = extract_types(noun, part_response, max_tokens, model, client)

    return part_dict, part_response, renamed_type_trace, excluded_type_trace


def ensure_part_format(noun, max_tokens, model):
    prompt = PART_PROMPT.replace("{noun}", noun)

    response = prompt_response(prompt, max_tokens, model)
    response = \
        response.split("Subtypes:", 1)[-1].split("Subtypes 6:", 1)[
            -1].split("Entity: ", 1)[-1].split("Entity 6: ", 1)[
            -1].split("Entity 7:", 1)[0].split("\nNote: ", 1)[0].split("- Note: ", 1)[0].strip()

    while has_subsubsubtypes(response) or has_wrong_format(response):
        response = prompt_response(prompt, max_tokens, model)
        response = \
            response.split("Subtypes:", 1)[-1].split("Subtypes 6:", 1)[
                -1].split("Entity: ", 1)[-1].split("Entity 6: ", 1)[
                -1].split("Entity 7:", 1)[0].split("\nNote: ", 1)[0].split("- Note: ", 1)[0].strip()

    return response


def ensure_part_semantics(noun, part_response, prompt, second_prompt, max_tokens, model):
    repeated_parts = get_repeated_parts(part_response)
    wrong_parts = get_wrong_parts(part_response)
    while repeated_parts or wrong_parts:
        # 1) Repeated parts: at least two types share the same parts in the response
        # 2) Mechanism parts: a part name ends with 'mechanism' or 'system', except 'internal mechanism'
        # 3) Additional parts: a part name contains 'additional' or 'various'
        # 4) Long parts: a part name appears to be too long
        # In these cases, we confirm the validity of the response by asking an additional question.
        for problem_prompt in [repeated_parts, wrong_parts]:
            if problem_prompt:
                part_response = ask_parts_again(noun, part_response, problem_prompt,
                                                prompt, second_prompt, max_tokens, model)
        repeated_parts = get_repeated_parts(part_response)
        wrong_parts = get_wrong_parts(part_response)

    return part_response


def get_repeated_parts(response):
    if 'No distinct subtypes' in response:
        return ''

    part_dict = dict()
    for line in response.strip().split("\n"):
        if line and line.strip()[0].isdigit() and ': ' in line:
            if re.match(r'( *)[0-9]+\.[a-z]\) (.*)', line):
                # subtypes
                line_number = line.strip().split(")", 1)[0]
            else:
                line_number = line.strip().split(". ", 1)[0]

            part_dict[line_number] = list()
            for part in line.split(":", 1)[-1].split(","):
                part_dict[line_number].append(part.split("(", 1)[0].strip())

    for k1, parts1 in part_dict.items():
        for k2, parts2 in part_dict.items():
            if k1 != k2 and set(parts1) == set(parts2):
                return f'Each type must be distinguished solely by the unique presence of their essential parts or components, but in your response "{", ".join(parts1)}" is repeated.'

    return ''


def get_wrong_parts(response):
    for line in response.strip().split("\n"):
        line = re.sub(r'(\s*)\([^)]*\)( *)', ' ', line).strip()
        if (line and line.strip()[0].isdigit() and ': ' in line) or ('Physical parts: ' in line):
            for part in re.split(' and | or |, ', line.split(": ", 1)[1]):
                # Part name contains 'mechanism' or 'system'; e.g., operating mechanism, trigger mechanism,
                # opening mechanism, scissor mechanism, heating system, etc.
                if (part.lower().endswith('mechanism') and 'internal mechanism' not in part.lower()) or\
                        (part.lower().endswith('system')):
                    return f"""The part "{part.strip()}" doesn't seem to refer to a tangible part."""
                # Part name contains 'additional' or 'various';
                # e.g., additional features for custom recipes, various integrated tools, etc.
                if 'additional' in part.lower() or 'various' in part.lower():
                    return 'Try to avoid using "additional" and "various" in your response.'
                # Part name is too long;
                # e.g., "combination of features from other subtypes" is too long to be a part's name
                # or modifiers like "usually at flush or near-flush level with the water body" can be filtered out.
                if len(part.split(" ")) > 5:
                    return f'The part "{part.strip()}" is too long.'
    return ''


def ask_parts_again(noun, prev_response, problem_prompt, prompt, second_prompt, max_tokens, model):
    # Ask subtypes and parts again by pointing out a problem in the previous response;
    # e.g., repeated parts, parts including "mechanism" or "system", or long parts.
    part_check_prompt = problem_prompt + "\n\n" + second_prompt.replace("{noun}", noun)
    message_list = [{"role": "user", "content": prompt.replace("{noun}", noun)},
                    {"role": "assistant", "content": prev_response},
                    {"role": "user", "content": part_check_prompt}]

    response = prompt_response(message_list, max_tokens, model)
    response = \
        response.split("Subtypes:", 1)[-1].split("Subtypes 6:", 1)[
            -1].split("Entity: ", 1)[-1].split("Entity 6: ", 1)[
            -1].split("Entity 7:", 1)[0].split("\nNote: ", 1)[0].split("- Note: ", 1)[0].strip()

    while has_subsubsubtypes(response) or has_wrong_format(response):
        response = prompt_response(message_list, max_tokens, model)
        response = \
            response.split("Subtypes:", 1)[-1].split("Subtypes 6:", 1)[
                -1].split("Entity: ", 1)[-1].split("Entity 6: ", 1)[
                -1].split("Entity 7:", 1)[0].split("\nNote: ", 1)[0].split("- Note: ", 1)[0].strip()

    return response


def has_subsubsubtypes(content):
    return any(re.match(r'( *)[0-9]+\.[a-z]\) (.*)', line) and (": " not in line) for line in content.split("\n"))


def has_wrong_format(content):
    content = content.rstrip('##')
    if 'No distinct subtypes' in content:
        # all non-empty lines should start with either 'No distinct' or 'Physical parts'
        return any(line.strip() and
                   not (line.strip().startswith('No distinct') or
                    line.strip().startswith('Physical parts:')) for line in content.split("\n"))
    else:
        # all non-empty lines should start with numbers only
        return any(line.strip() and not line.strip()[0].isdigit() for line in content.split("\n"))


def extract_types(noun, response, max_tokens, model, client):
    part_dict = dict()
    if "No distinct subtypes" in response:
        # no subtypes and no subsubtypes
        return {'-': {'-': response.split("\nPhysical parts:")[-1].rstrip('##')}}, "", ""
    else:
        for line in response.split("\n"):
            if line.strip() and line.strip()[0].isdigit():
                if re.search(r'\.[a-z]\)', line):
                    # subsubtype information
                    subsubtype, parts = line.split(") ", 1)[-1].split(": ", 1)
                    if not parts.strip():
                        return dict()
                else:
                    # subtype information
                    subsubtype = "-"
                    num, line_content = line.split(". ", 1)
                    if (": " in line.strip()) and (num+".a)" not in response):
                        subtype, parts = line_content.split(": ", 1)
                    else:
                        subtype = line_content.split(":")[0]
                        continue  # move onto extracting subsubtypes

                if subtype and (subtype not in part_dict.keys()):
                    part_dict[subtype] = dict()

                part_dict[subtype][subsubtype] = parts.rstrip('##')

        if len(part_dict.keys()) == 1:
            if len(part_dict[subtype].keys()) == 1:
                part_dict = {'-': {'-': part_dict[subtype][subsubtype]}}
            else:
                temp_part_dict = dict()
                for subsubtype, subsub_parts in part_dict[subtype].items():
                    temp_part_dict[subsubtype] = subsub_parts
                part_dict = temp_part_dict

        # rename sub- and subsub-types
        confirmed_part_dict = dict()
        renamed_type_trace = ""
        excluded_type_trace = ""
        for subtype, subpart_dict in part_dict.items():
            if '(optional)' in subtype:
                continue
            new_subtype = confirm_subtype(subtype, noun, 512, 128, model, client)
            if new_subtype:
                if new_subtype.lower() in [lower_type.lower() for lower_type in confirmed_part_dict.keys()] +\
                        [re.sub(r' \([^)]*\)$', '', lower_type.lower()) for lower_type in confirmed_part_dict.keys()]:
                    # If the new name for the subtype already exists as another subtype, find subtypes again.
                    return dict(), "", ""
                if new_subtype.lower() == re.sub(r' \([^)]*\)$', '', noun.lower()) or\
                        new_subtype.lower() in [subsubtype.lower() for subsubtype in subpart_dict.keys()] +\
                        [re.sub(r' \([^)]*\)$', '', subsubtype.lower()) for subsubtype in subpart_dict.keys()]:
                    # If the new name for the subtype is the same as its supertype name
                    # or the new name already exists as one of its subsubtypes, revert it back.
                    new_subtype = subtype
                if new_subtype.lower() != subtype.lower():
                    renamed_type_trace += f"{subtype} is renamed as {new_subtype}.\n"
                if new_subtype.endswith(')'):
                    within_paren = new_subtype.rsplit("(", 1)[1].strip(")")
                    if " " in within_paren and all(token[0].isupper() for token in within_paren.split(" ")):
                        # take the content within parentheses
                        # e.g., take 'Organic Light Emitting Diodes' instead of 'OLED'
                        new_subtype = within_paren
                    else:
                        # remove parentheses at the end
                        new_subtype = re.sub(r' \([^)]*\)$', '', new_subtype.strip("."))

                confirmed_part_dict[new_subtype] = dict()
                for subsubtype, parts in subpart_dict.items():
                    if '(optional)' in subsubtype:
                        continue
                    new_subsubtype = confirm_subtype(subsubtype, new_subtype, 512, 128, model, client)
                    if new_subsubtype:
                        if new_subsubtype.lower() in [lower_type.lower() for lower_type in confirmed_part_dict[new_subtype].keys()]:
                            # If the new name for the subsubtype already exists as another subsubtype, find subsubtypes again.
                            return dict(), "", ""
                        if new_subsubtype.lower() == new_subtype.lower() or \
                                new_subsubtype.lower() == re.sub(r' \([^)]*\)$', '', noun.lower()):
                            # If the new name for the subsubtype is the same as its subtype name, revert it back.
                            new_subsubtype = subsubtype
                        if new_subsubtype.lower() != subsubtype.lower():
                            renamed_type_trace += f"{subsubtype} is renamed as {new_subsubtype}.\n"
                        if new_subsubtype.endswith(')'):
                            within_paren = new_subsubtype.rsplit("(", 1)[1].strip(")")
                            if " " in within_paren and all(token[0].isupper() for token in within_paren.split(" ")):
                                # take the content within parentheses
                                new_subsubtype = within_paren
                            else:
                                # remove parentheses at the end
                                new_subsubtype = re.sub(r' \([^)]*\)$', '', new_subsubtype.strip("."))
                        confirmed_part_dict[new_subtype][new_subsubtype] = parts
                    else:
                        excluded_type_trace += f"{subsubtype} is removed from the subtypes of {new_subtype}.\n"
                if len(confirmed_part_dict[new_subtype].keys()) == 0:
                    # when a subtype exists with no valid subsubtypes,
                    # but the part information we have is about subsubtypes,
                    # we need to re-extract subsubtypes, so we run the whole type/part prompt again
                    return dict(), "", ""
            else:
                excluded_type_trace += f"{subtype} is removed from the subtypes of {noun}.\n"

        return confirmed_part_dict, renamed_type_trace, excluded_type_trace


def ask_materials(noun, part_dict, result_df, col_names, max_tokens=MAX_TOKENS, model=MODEL, prompt=MATERIAL_PROMPT):
    simple_noun = noun.split(" (", 1)[0]
    if " (" in noun:
        category = " (" + noun.split(" (")[-1]
    else:
        category = ""

    for subtype, subsub_info in part_dict.items():
        if subtype == "-":
            item = noun
        else:
            item = re.sub(r'\([^)]*\)( *)', '', subtype).strip() + category

        for subsubtype, parts in subsub_info.items():
            if subsubtype != "-":
                item = re.sub(r'\([^)]*\)( *)', '', subsubtype).strip() + category

            parts_list = re.split(r",\s*(?![^()]*\))|\n- ", "\n" + parts)
            # convert 'optional'+[part] name to the optionality [part]:T
            parts_optional = {re.sub(r'^(optional) ', '', re.sub(r'\([^)]*\)( *)', '', part).strip(), flags=re.IGNORECASE).strip(): (
                    '(optional' in part or part.lower().startswith('optional '))
                    for part in parts_list if re.sub(r'\([^)]*\)( *)', '', part).strip()}

            if len(parts_optional.keys()) == 1:
                parts_str = "-"  # no distinct parts
                parts_optionality = "-: F"
                if (simple_noun.lower() not in item.lower()) and (not category):
                    # when there are no specified parts, the subtype or subsubtype's name may be confusing,
                    # so we add the noun within parentheses as a guidance, unless the name doesn't indicate the noun;
                    # e.g., Coat of arms -> Helmet (shouldn't be a general helmet)
                    # e.g., Baby sling -> Wrap (shouldn't be a general wrap which can be made of paper)
                    # e.g., Coffee filter -> Disk coffee filter (noun is already indicated in the subtype)
                    item += " (" + simple_noun.lower() + ")"
            else:
                # Indicate optionality (T/F) for each part.
                # If a part name is surrounded by quotation marks, remove them.
                parts_str = ", ".join([re.sub(r"(^\")|(\"$)", "", part) if part.startswith('"') and part.endswith('"')
                                       else part for part in parts_optional.keys()])
                parts_optionality = "\n".join([re.sub(r"(^\")|(\"$)", "", p) + ": " + str(o)[0] if p.startswith('"') and p.endswith('"')
                                               else p + ": " + str(o)[0] for p, o in parts_optional.items()])

            question = prompt.replace("{noun}", item).replace("{parts}", parts_str)
            response = prompt_response(question, max_tokens, model)

            response, parts_optionality = ensure_material_semantics(response, question, parts_optionality, max_tokens, model)
            response = response.split("Entity: ")[-1].split("Entity 6:")[-1].split("Materials:", -1)[-1].strip()

            part_count = len(parts_optionality.split("\n"))
            material_count = len(response.split("\n"))
            part_conj_count = parts_optionality.count(' or ') + parts_optionality.count(' and ') + parts_optionality.count(' and/or ')
            if (part_count + part_conj_count) != material_count:
                # If the number of parts don't match the number of parts under materials,
                # e.g., "long, narrow blade" is parsed as "long" (part1) and "narrow blade" (part2),
                # re-run the prompts from the beginning.
                # It counts the number of parts plus the number of conjunctions
                # and compare the numbers to the part count under materials, so that
                # e.g., when a part name is "valves or keys" and under materials,
                # each of "vales" and "keys" gets their own material response, that's fine.
                return result_df, True

            result_df = append_rows(result_df, col_names, [noun, subtype, subsubtype, parts_optionality, response])

    return result_df, False


def ensure_material_semantics(response, prompt, parts_optionality, max_tokens, model):
    if materials_too_long(response) or ('unknown' in response.lower()) or verbose_materials(response):
        # Prompt the model for the composition materials again
        response = prompt_response(prompt, max_tokens, model)
        while verbose_materials(response):
            response = prompt_response(prompt, max_tokens, model)
        if materials_too_long(response) or 'unknown' in response.lower():
            response = prompt_response(prompt, max_tokens, model)
            while verbose_materials(response):
                response = prompt_response(prompt, max_tokens, model)
            if materials_too_long(response) or 'unknown' in response.lower():
                # If the third response is suspiciously long as well, then the item is considered as
                # a dependent component that doesn't have its own materials (e.g., watermark),
                # an unknown object (e.g., Unidentified flying object),
                # or a type that has very wide varieties of subtypes (e.g., used good, work of art)
                parts_optionality = "-"
                response = "-"
    return response, parts_optionality


def materials_too_long(response):
    # When a response doesn't list parts and is long, the noun it's describing tends to be
    # something unreal or cannot stand on its own.
    if response.count('Entity 6:') > 1:
        return True
    response = response.split("Entity: ")[-1].split("Entity 6:")[-1].split("Materials:", -1)[-1].strip()
    if '1. ' not in response:
        for line in response.split("\n"):
            line = re.sub(r'\([^)]*\)( *)', '', line).strip()
            material_list = re.split(r' and\/or | and | or |,|, ', line.split(": ", 1)[-1])
            for material in material_list:
                if len(material.strip().split(" ")) > 5:
                    return True
    return False


def clean_parts_materials(raw_df, col_names):
    clean_df = pd.DataFrame(columns=col_names)
    for ind, row in raw_df.iterrows():
        parts_optionality = row['optionality']
        materials = row['response']

        if materials.count("\n") == 0:
            materials = extract_material_str(materials.strip("."))
        else:
            part_count = 1
            for part_mtrl in copy.deepcopy(materials).split("\n"):
                part, material = part_mtrl.split(". ", )[1].split(": ", 1)
                # remove quotation marks around the part name
                if part.startswith('"') and part.endswith('"'):
                    part = re.sub(r"(^\")|(\"$)", "", part)
                # remove asterisks and parentheses as well
                part = re.sub(r"\*", "", part)
                part = re.sub(r' \([^)]*\)', '', part).strip('. ')
                material_str = part_mtrl.strip().split(": ", 1)[1]

                # find if the material is the same as another part's.
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
                                    new_material_str += ', ' + (', ' + new_material_conj + ' ').join(temp_material_str.rsplit(', ', 1))
                            else:
                                new_material_str = twin_candidates[0]
                    material_str = new_material_str

                else:
                    if 'same ' in material_str.lower() and \
                            any(delimiter in material_str for delimiter in {'(', '-', '–', ':', ';'}):
                        material_str = re.split(r'\(|-|–|:|;',
                                                re.split('same ', material_str, re.IGNORECASE)[1])[-1].strip("). ")

                # cleaned material information without the part number
                material_str = extract_material_str(material_str)

                if material_str == '-':
                    # non-physical materials; remove the corresponding part
                    if materials.startswith(part_mtrl):
                        materials = materials.replace(part_mtrl, '', 1).strip()
                    else:
                        materials = materials.replace("\n" + part_mtrl, '').strip()
                    parts_optionality = re.sub(f'(^|\n){part}: (T|F)', '', parts_optionality).strip()
                else:
                    # Every new line starts with a number indicating a part.
                    materials = materials.replace(part_mtrl, str(part_count) + ". " + part + ": " + material_str)
                    part_count += 1

            # parts reduced to no distinct parts after removing non-material responses
            if parts_optionality.count("\n") == 0:
                parts_optionality = "-: F"
                materials = materials.split(": ", 1)[1]

        if not materials or materials == '-':
            parts_optionality, materials = '-', '-'

        # change 'electronics' material to 'electronic materials'
        new_materials = re.sub(r':([^\n]+)electronics', ':\\1electronic materials', materials)
        clean_df = append_rows(clean_df, col_names,
                               [row['item'], row['subtype'], row['subsubtype'], parts_optionality, new_materials])
    return clean_df


def main():
    with open(NOUN_FILE, 'r') as f_read:
        col_names = ['item', 'subtype', 'subsubtype', 'optionality', 'response']
        result_df = pd.DataFrame(columns=col_names)
        noun = f_read.readline().strip()
        while noun:
            part_material_unmatch = True
            while part_material_unmatch:
                # Ask about subtypes and parts
                part_dict, part_response, renamed_type_trace, excluded_type_trace = ask_type_parts(noun)
                # Ask about materials
                temp_result_df, part_material_unmatch = ask_materials(noun, part_dict, result_df, col_names)

            # leave a quick progress log
            f_write = open(LOG_FILE, 'a')
            f_write.write(f"Noun: {noun}\n")
            f_write.write(part_response.strip() + "\n")
            if renamed_type_trace:
                f_write.write("Renamed:\n" + renamed_type_trace)
            if excluded_type_trace:
                f_write.write("Deleted:\n" + excluded_type_trace)
            f_write.write("\n")
            f_write.close()

            result_df = temp_result_df
            result_df.to_csv(RESULT_FILE, sep='\t', index=False, encoding='utf-8')

            noun = f_read.readline().strip()

        clean_df = clean_parts_materials(result_df, col_names)
        clean_df.to_csv(CLEAN_RESULT_FILE, sep='\t', index=False, encoding='utf-8')


if __name__ == "__main__":
    main()
