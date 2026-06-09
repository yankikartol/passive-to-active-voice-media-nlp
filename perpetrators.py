import json
import os
import re
import matplotlib
import matplotlib.pyplot as plt
import spacy
from wordcloud import WordCloud, STOPWORDS

# Ensure matplotlib works seamlessly in headless/server environments (e.g., GitHub Actions)
matplotlib.use('Agg')

# Load the Spacy English NLP model globally to optimize performance
try:
    nlp_english = spacy.load("en_core_web_sm")
except OSError:
    print("[ERROR] Please install the English language model: python -m spacy download en_core_web_sm")


def load_or_create_dataset(filename="headlines.json"):
    """
    Loads the dataset from a JSON file.
    If the file does not exist, automatically creates a new one with default sample data.
    """
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Fallback sample dataset if the JSON file is missing
    default_dataset = [
        {"turkish_raw": "Eski eşi tarafından katledildi.", "english_translated": "She was slaughtered by her ex-husband."},
        {"turkish_raw": "Genç kadın otel odasında bıçakla katledildi.", "english_translated": "The young woman was stabbed to death in her hotel room."},
        {"turkish_raw": "Bir kadın daha eşi tarafından öldürüldü.", "english_translated": "Another woman was killed by her husband."},
        {"turkish_raw": "Genç bir kadın daha eşi olacak cani tarafından öldürüldü.", "english_translated": "Yet another young woman has been killed by the monster who was to become her husband."},
        {"turkish_raw": "Tren istasyonunda katletildi.", "english_translated": "She was killed at the train station."},
        {"turkish_raw": "Kadınlar Günü’nde katledildi.", "english_translated": "A woman was killed on Women's Day."},
        {"turkish_raw": "Karısını iple boğarak öldürdü.", "english_translated": "He strangled his wife with a rope."},
        {"turkish_raw": "2 kadın öldürüp ormana gömmüşler.", "english_translated": "They killed two women and buried them in the forest."},
        {"turkish_raw": "Eşi tarafından katledildi.", "english_translated": "She was killed by her husband."},
        {"turkish_raw": "Boşanma aşamasındaki eşini vurup polise teslim oldu.", "english_translated": "He shot his wife during their divorce proceedings and turned himself in to the police."},
        {"turkish_raw": "2 çocuk annesi kadın evinde bıçaklanarak öldürüldü.", "english_translated": "Two children's mother was stabbed to death in her house."},
        {"turkish_raw": "Çocuklarının gözü önünde öldürdü.", "english_translated": "She was killed in front of her children."},
        {"turkish_raw": "Nisan ayında 49 kadın öldürüldü.", "english_translated": "49 women murdered in April."},
        {"turkish_raw": "Iki kadın aynı evde öldürüldü.", "english_translated": "Two women killed in the same house."}
    ]
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(default_dataset, f, ensure_ascii=False, indent=4)
    print(f"[INFO] '{filename}' not found. Created a new file with sample data.")
    return default_dataset


def clean_typography(text):
    """
    Cleans typographical anomalies, extra spaces, and whitespace 
    before punctuation using regex.
    """
    text = re.sub(r'\s+([.,!?;])', r'\1', text) 
    text = re.sub(r'\s*-\s*', '-', text)       
    text = re.sub(r"\s+('s)\b", r"\1", text)
    text = re.sub(r'\s+', ' ', text).strip()
    if not text.endswith("."):
        text += "."
    return text


def convert_passive_to_active(english_headline, pronoun_mapping, residual_words, violence_verbs):
    """
    Analyzes the NLP Dependency Tree to convert a passive headline 
    into an active sentence structure, returning the rewritten text and the perpetrator.
    """
    doc = nlp_english(english_headline)
    
    # Primary check for formal passive voice tags
    is_passive = any(token.dep_ in ("nsubjpass", "auxpass") for token in doc)
    has_violence_verb = any(t.text.lower() in violence_verbs for t in doc)
    
    # Catch news headline short-passive structures (e.g., "49 women murdered")
    if not is_passive and has_violence_verb:
        for token in doc:
            if token.text.lower() in violence_verbs and token.tag_ == "VBN":
                is_passive = True
                break

    # --- PASSIVE VOICE ANALYSIS ---
    verb_token = None
    victim_words_list = []
    explicit_perpetrator_tokens = []
    victim_head_token = None

    for token in doc:
        # Identify the true syntactic victim/subject of the passive voice
        if token.dep_ in ("nsubjpass", "nsubj") and not victim_words_list:
            victim_head_token = token
            victim_words_list = [t.text.lower() for t in token.subtree if t.text.lower() not in {",", "."}]
            victim_phrase_raw = " ".join([t.text for t in token.subtree if t.text not in {",", "."}])

        if has_violence_verb:
            if token.text.lower() in violence_verbs:
                verb_token = token
        else:
            if token.dep_ == "auxpass":
                verb_token = token.head

        if token.dep_ == "agent":
            explicit_perpetrator_tokens = [t.text for t in token.subtree if t.text.lower() != "by"]

    # Fallback mechanisms if SpaCy fails to tag the subject cleanly (e.g., short headlines)
    if not victim_words_list:
        potential_victims = [t for t in doc if t.pos_ in ("NOUN", "PROPN", "PRON", "NUM")]
        if potential_victims:
            victim_head_token = potential_victims[0]
            victim_words_list = [t.text.lower() for t in victim_head_token.subtree if t.text.lower() not in {",", "."}]
            victim_phrase_raw = " ".join([t.text for t in victim_head_token.subtree if t.text not in {",", "."}])
        else:
            victim_head_token = doc[0]
            victim_words_list = [doc[0].text.lower()]
            victim_phrase_raw = doc[0].text
    
    if not verb_token:
        for token in doc:
            if token.text.lower() in violence_verbs:
                verb_token = token
                break

    # --- RIGOROUS PLURAL VICTIM DETECTION ---
    is_plural_victim = False
    if victim_head_token and (victim_head_token.tag_ in ("NNS", "NNPS") or victim_head_token.text.lower() == "women"):
        is_plural_victim = True
        
    clean_text_lower = english_headline.lower()
    if not is_plural_victim:
        if "women" in clean_text_lower and "women's day" not in clean_text_lower:
            if not re.search(r"\b(children|women)'s\s+(mother|wife|sister|daughter)\b", clean_text_lower):
                is_plural_victim = True

    default_perpetrator = "Men" if is_plural_victim else "A man"

    # --- CASE 1: THE HEADLINE IS ALREADY IN ACTIVE VOICE ---
    if not is_passive:
        if doc[0].text.lower() == "he":
            corrected_text = "A man " + " ".join([t.text for t in doc[1:]])
            return corrected_text, "A man"
        elif doc[0].text.lower() == "they":
            corrected_text = "The suspects " + " ".join([t.text for t in doc[1:]])
            return corrected_text, "The suspects"
        else:
            perpetrator = doc[0].text.capitalize() if doc[0].pos_ in ("NOUN", "PROPN") else default_perpetrator
            return english_headline, perpetrator

    # --- CASE 2: THE HEADLINE IS IN PASSIVE VOICE ---
    if verb_token:
        action = verb_token.text.lower()
        context_words = []
        perpetrator_lower = [w.lower() for w in explicit_perpetrator_tokens]

        for token in doc:
            t_lower = token.text.lower()
            if (token.dep_ != "auxpass" and 
                token != verb_token and 
                t_lower not in residual_words and 
                t_lower not in victim_words_list and 
                t_lower not in perpetrator_lower and 
                t_lower != "by" and
                token.pos_ != "PUNCT"):
                context_words.append(token.text)

        context_phrase = " ".join(context_words)
        victim_lower = " ".join(victim_words_list).strip()
        victim_phrase = victim_phrase_raw

        first_token = [t for t in doc if t.text.lower() in victim_words_list]
        if first_token and first_token[0].pos_ != "PROPN" and victim_phrase:
            victim_phrase = victim_phrase[0].lower() + victim_phrase[1:]

        # MAP PRONOUNS EXPLICITLY TO "the woman" OR OTHER TARGETS TO EMPOWER THE ACTIVE VOICE
        if victim_lower in pronoun_mapping:
            victim_phrase = pronoun_mapping[victim_lower]
        elif victim_lower.startswith(("the ", "a ", "an ")):
            victim_phrase = "the " + victim_phrase_raw.split(" ", 1)[1]

        if "woman" in victim_words_list and not is_plural_victim:
            victim_phrase = "the young woman" if "young" in victim_words_list else "the woman"

        if explicit_perpetrator_tokens:
            perpetrator_phrase = " ".join(explicit_perpetrator_tokens)
            perpetrator = perpetrator_phrase[0].upper() + perpetrator_phrase[1:]
        else:
            perpetrator = default_perpetrator

        if context_phrase.strip():
            corrected_text = f" {perpetrator} {action} {victim_phrase} {context_phrase}"
        else:
            corrected_text = f" {perpetrator} {action} {victim_phrase}"
            
        return corrected_text, perpetrator
    
    return english_headline, default_perpetrator


def generate_perpetrators_wordcloud(perpetrators, output_dir="output"):
    """
    Generates a WordCloud from the extracted perpetrator list and saves it to the specified directory.
    """
    if not perpetrators:
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    text_for_cloud = " ".join(perpetrators)
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update(["become", "was", "to"])

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        min_font_size=10,
        colormap='Reds',
        stopwords=custom_stopwords
    ).generate(text_for_cloud)

    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    
    save_path = os.path.join(output_dir, "perpetrators_wordcloud.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Word cloud successfully saved to: {os.path.abspath(save_path)}")


def main():
    """
    Main execution pipeline. Initializes configurations and orchestrates the process.
    """
    print("\n" + "="*20 + " Passive Voice Analysis and Correction " + "="*20 + "\n")

    # Mapping pronouns directly to explicit nouns to keep the focus on the human element
    pronoun_mapping = {
        "she": "the woman", 
        "her": "the woman", 
        "they": "the women", 
        "them": "the women",
        "i": "me", 
        "we": "us"
    }
    
    residual_words = {"was", "were", "is", "are", "being", "been", "have", "has", "had", "be", "yet", "another"}
    violence_verbs = {"killed", "slaughtered", "stabbed", "assaulted", "murdered", "shot", "strangled"}

    dataset = load_or_create_dataset("headlines.json")
    all_perpetrators = []

    for item in dataset:
        english_headline = item["english_translated"]
        
        raw_corrected, perpetrator = convert_passive_to_active(
            english_headline, pronoun_mapping, residual_words, violence_verbs
        )
        
        final_corrected = clean_typography(raw_corrected)
        all_perpetrators.append(perpetrator)
        
        print(f"Original English : {english_headline}")
        print(f"Corrected Active : {final_corrected}\n")

    generate_perpetrators_wordcloud(all_perpetrators)


if __name__ == "__main__":
    main()