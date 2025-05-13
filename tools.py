import requests

def get_definition(word):
    """
    Fetch the dictionary definition of a word using dictionaryapi.dev
    """
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and "meanings" in data[0]:
            meanings = data[0]["meanings"]
            definitions = []
            for meaning in meanings:
                for definition in meaning["definitions"]:
                    definitions.append(definition["definition"])
            return f"Definitions of '{word}':\n" + "\n".join(f"- {d}" for d in definitions[:3])
        else:
            return f"No definition found for '{word}'."
    else:
        return f"Failed to fetch definition for '{word}'."
