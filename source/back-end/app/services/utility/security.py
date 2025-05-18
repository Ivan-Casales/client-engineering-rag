import string

def sanitize_input(text: str, max_length: int = 512) -> str:
    printable = set(string.printable)
    cleaned = ''.join(ch for ch in text if ch in printable)

    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]

    cleaned = cleaned.replace('{', '{{').replace('}', '}}')

    return cleaned.strip()
