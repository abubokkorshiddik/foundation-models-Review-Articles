
import os
import time
import pandas as pd
from datetime import datetime
from Bio import Entrez
from habanero import Crossref

# ---------------- CONFIG ----------------
Entrez.email = "test@test.com"
cr = Crossref(mailto="test@test.com")

OUT_DIR = "sr_batches"
os.makedirs(OUT_DIR, exist_ok=True)

YEAR_FROM = 2015
YEAR_TO = 2026

BATCH_SIZE = 400
CHUNK_SIZE = 80
SLEEP_TIME = 0.2

# ---------------- SEARCH TERMS ----------------
AI_TERMS = [
    '"foundation model"',
    '"large language model"',
    'LLM',
    'GPT',
    'BERT',
    'transformer',
    'multimodal'
]

HEALTH_TERMS = [
    '"public health"',
    'epidemiology',
    'disease',
    'surveillance',
    'prediction',
    'outbreak'
]

# ---------------- HELPERS ----------------
def build_query(ai):
    return f'({ai}) AND ({" OR ".join(HEALTH_TERMS)}) AND ("{YEAR_FROM}"[DP] : "{YEAR_TO}"[DP])'

def extract_model(text):
    text = text.lower()
    models = []
    for k in ["gpt", "bert", "transformer", "llm", "multimodal"]:
        if k in text:
            models.append(k)
    return ", ".join(set(models)) if models else "Not specified"

def extract_limitations(text):
    sentences = text.lower().split(".")
    lim = [s.strip() for s in sentences if any(w in s for w in ["limitation", "challenge", "bias"])]
    return " | ".join(lim[:2]) if lim else "Not stated"

# ---------------- FAST PUBMED ----------------
def pubmed_batch_fast(run_id=2):
    all_data = []

    for ai in AI_TERMS:
        query = build_query(ai)

        handle = Entrez.esearch(db="pubmed", term=query, retmax=3000)
        record = Entrez.read(handle)
        ids = record["IdList"]

        start = run_id * BATCH_SIZE
        end = start + BATCH_SIZE
        ids = ids[start:end]

        print(f"{ai} → {len(ids)} records (FAST batch {run_id})")

        for i in range(0, len(ids), CHUNK_SIZE):
            batch_ids = ids[i:i+CHUNK_SIZE]

            try:
                fetch = Entrez.efetch(
                    db="pubmed",
                    id=",".join(batch_ids),
                    rettype="medline",
                    retmode="text"
                )

                data = fetch.read()

                papers = data.split("\n\nPMID- ")

                for paper in papers:
                    title, abstract, doi = "", "", ""

                    for line in paper.split("\n"):
                        if line.startswith("TI  -"):
                            title = line.replace("TI  -", "").strip()
                        elif line.startswith("AB  -"):
                            abstract = line.replace("AB  -", "").strip()
                        elif "doi" in line.lower():
                            doi = line.split()[-1]

                    if title.strip() == "":
                        continue

                    text = title + " " + abstract

                    all_data.append({
                        "Title": title,
                        "Abstract": abstract,
                        "Model_Used": extract_model(text),
                        "Limitations": extract_limitations(text),
                        "DOI": doi,
                        "Source": "PubMed",
                        "Link": f"https://pubmed.ncbi.nlm.nih.gov/{batch_ids[0]}/"
                    })

                time.sleep(SLEEP_TIME)

            except:
                continue

    return all_data

# ---------------- CROSSREF ----------------
def crossref_batch(run_id=2):
    all_data = []

    for ai in AI_TERMS:
        query = build_query(ai)

        try:
            works = cr.works(query=query, rows=1000)
            items = works["message"]["items"]

            start = run_id * BATCH_SIZE
            end = start + BATCH_SIZE
            items = items[start:end]

            for item in items:
                title = item.get("title", [""])[0]
                abstract = item.get("abstract", "")
                doi = item.get("DOI")

                text = title + str(abstract)

                all_data.append({
                    "Title": title,
                    "Abstract": abstract,
                    "Model_Used": extract_model(text),
                    "Limitations": extract_limitations(text),
                    "DOI": doi,
                    "Source": "CrossRef",
                    "Link": f"https://doi.org/{doi}" if doi else None
                })

        except:
            continue

    return all_data

# ---------------- MAIN ----------------
def main(run_id=2):
    print(f"\n=== RUN {run_id} (FAST MODE) ===")

    pubmed_data = pubmed_batch_fast(run_id)
    crossref_data = crossref_batch(run_id)

    df = pd.DataFrame(pubmed_data + crossref_data)

    print("Collected:", len(df))

    df = df.drop_duplicates(subset=["Title"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(OUT_DIR, f"batch_{run_id}_{timestamp}.csv")

    df.to_csv(file_path, index=False)

    print("Saved:", file_path)

# ---------------- RUN ----------------
if __name__ == "__main__":
    main(run_id=2)





    import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Data: Number of studies per model per domain
# -------------------------
data = {
    'Application domain': [
        'NLP / Text mining',
        'Clinical QA / Decision support',
        'Diagnosis / Risk prediction',
        'Imaging / Vision tasks',
        'Outbreak / Environmental monitoring'
    ],
    'BERT': [8, 0, 0, 0, 0],
    'GPT': [0, 12, 0, 0, 0],
    'LLM': [0, 0, 28, 0, 0],
    'Multimodal': [0, 0, 0, 0, 14],
    'Transformer': [0, 0, 0, 20, 0]
}

df = pd.DataFrame(data)
df.set_index('Application domain', inplace=True)

# -------------------------
# Plot stacked bar chart
# -------------------------
fig, ax = plt.subplots(figsize=(10,6))

df.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')

# Labels and title
ax.set_ylabel('Number of studies')
ax.set_xlabel('Application domain')
ax.set_title('Model-specific applications and domains of foundation models', fontsize=14)
ax.legend(title='Model type', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
