def create_llama_prompt(
    input_text, target_text=None, is_training=False, eos_token="</s>"
):
    if target_text is None:
        return f"{input_text}{eos_token if is_training else ''}"
    else:
        return f"[INST] Do not define a function. Do not import anything. Do not write any comments. Generate one line of Python code snippet to satisfy the following description: {input_text}. [/INST] CODE: {target_text}</s>"

def get_humaneval_prompter(mode=None):
    match mode:
        case "a":
            def promptera(prompt):
                return f"Please write a Python code snippet that satisfies the following description: {prompt}"
            return promptera
        case _:
            def prompter(prompt):
                return prompt
            return prompter



def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")


def clean_whitespaces_generations(text):
    trim_list = [" ", "\n"]
    trim_map = {" ": "sp", "\n": "nl"}
    new_text = text[0]
    last_ch = text[0]
    occ = 0
    for ch in text[1:]:
        if last_ch in trim_list and ch != last_ch:
            if occ > 20:
                new_text += f"<{trim_map[last_ch]}{occ}>"
                occ = 0
            else:
                new_text += last_ch * (occ + 1)
                occ = 0
        if ch not in trim_list:
            new_text += ch
        else:
            if ch == last_ch:
                occ += 1
        last_ch = ch

    if last_ch in trim_list:
        if occ > 1:
            new_text += f"<{trim_map[last_ch]}{occ}>"
            occ = 0
        else:
            new_text += last_ch

    return new_text


def find_nth(haystack: str, needle: str, n: int) -> int:
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start + len(needle))
        n -= 1
    return start


def join_prefix_prompt(prefix, prompt):
    return f"{prefix}{prompt}"


def csn_create_prompt(input):
    return f"{input}\n# summary:\n"


def csn_join(prompt, completion):
    return f"{prompt}{completion}"


def csn_split(input):
    splits = input.split("# summary:")
    prompt = splits[0]
    if len(splits) == 1:
        return prompt, ""
    completion = splits[1]
    return prompt, completion

def spp_join(prompt, completion):
    return f"{prompt}{completion}"

def spp_split(input):
    idx_split = find_nth(input, '"""', 2) + 3
    if idx_split == -1:
        return input, ""
    prompt = input[:idx_split]
    completion = input[idx_split:]
    return prompt, completion
