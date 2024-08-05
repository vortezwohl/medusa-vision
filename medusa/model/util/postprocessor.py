def output_stringify(output: tuple, tag: str, top_k: int) -> str:
    return (
        output[:top_k].__str__()
        .replace('(', '')
        .replace(')', '')
        .replace('{', '')
        .replace('}', '')
        .replace(f"'{tag}':", '')
        .replace("'probability':", '')
        .replace("',", ':')
        .replace("'", '')
        .replace(' ', '')
        .replace(',', ' ')
    )
