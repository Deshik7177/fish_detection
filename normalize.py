SPECIES_MAP = {
    "acanthuridae": "surgeon",
    "surgeonfishes": "surgeon",
    "surgeon": "surgeon",

    "balistidae": "trigger",
    "triggerfishes": "trigger",
    "trigger": "trigger",

    "carangidae": "jack",
    "jacks": "jack",
    "jack": "jack",

    "ephippidae": "spade",
    "spadefishes": "spade",
    "spade": "spade",

    "labridae": "wrasse",
    "wrasse": "wrasse",

    "lutjanidae": "snapper",
    "snappers": "snapper",
    "snapper": "snapper",

    "pomacanthidae": "angel",
    "angelfishes": "angel",
    "angel": "angel",

    "pomacentridae": "damsel",
    "damselfishes": "damsel",
    "damsel": "damsel",

    "scaridae": "parrot",
    "parrotfishes": "parrot",
    "parrot": "parrot",

    "scombridae": "tuna",
    "tunas": "tuna",
    "tuna": "tuna",

    "serranidae": "grouper",
    "groupers": "grouper",
    "grouper": "grouper",

    "selachimorpha": "shark",
    "shark": "shark",

    "zanclidae": "moorish idol",
    "moorish": "moorish idol",
}

def normalize_species(raw_label):
    label = raw_label.lower().replace("-", " ").replace("(", "").replace(")", "")
    for word in label.split():
        if word in SPECIES_MAP:
            return SPECIES_MAP[word]
    return raw_label
