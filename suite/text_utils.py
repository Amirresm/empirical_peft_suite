def create_llama_prompt(
    input_text, target_text=None, is_training=False, eos_token="</s>"
):
    if target_text is None:
        return f"{input_text}{eos_token if is_training else ''}"
    else:
        return f"[INST] Do not define a function. Do not import anything. Do not write any comments. Generate one line of Python code snippet to satisfy the following description: {input_text}. [/INST] CODE: {target_text}</s>"


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
