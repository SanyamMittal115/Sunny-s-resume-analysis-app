from transformers import pipeline
from pypdf import PdfReader
import textwrap
import streamlit as st





chunk = 800
Target = ["Product Manager", "AI Engineer", "Business Analyst", "Solutions Engineer"]
uploadfil = st.file_uploader("Upload your resume in PDF format", type=["pdf"])
print("Loading model...")

classifymodel = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
summarymodel = pipeline("summarization",model="facebook/bart-large-cnn")
ner = pipeline("ner",model="dslim/bert-base-NER", aggregation_strategy="simple")
ner = pipeline("ner",model="dslim/bert-base-NER", aggregation_strategy="simple", device=-1, framework="pt")

def text_extract(path):
    reader = PdfReader(path)
    txt = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            txt += page_text + "\n"
    
    if not txt.strip():
        raise ValueError("PDF is not readable.")
    else:
        st.subheader("Extracted text")
        return txt
       
if uploadfil is not None:
    st.write("File upload:", uploadfil.name)
    
# Chunktext
def chunksplit(txt, max_length):
    chunks = []
    txtsum = ""

    for line in txt.split("\n"):
        if len(txtsum) + len(line) < max_length:
            txtsum += line + " "
        else:
            chunks.append(txtsum.strip())
            txtsum = line + " "

    if txtsum:
        chunks.append(txtsum.strip())

    return chunks

#Analysis
def analyze_role_fit(chunks):
    scores = {role: 0 for role in Target}

    for chunk in chunks:
        result = classifymodel(chunk, Target)
        for label, score in zip(result["labels"], result["scores"]):
            scores[label] += score

    # Normalize
    total = sum(scores.values())
    for role in scores:
        scores[role] = round(scores[role] / total, 3)

    return scores

#Extract skills
def extract_skills(chunks):
    skills = set()

    for chunk in chunks:
        entities = ner(chunk)
        for ent in entities:
            if ent["entity_group"] in ["ORG", "MISC", "PER"]:
                skills.add(ent["word"])

    return sorted(skills)

#Summary 
def summarize_resume(chunks):
    summaries = []

    for chunk in chunks[:3]: 
        summary = summarymodel(chunk, max_length =120, min_length =40, do_sample=False)[0]["summary_text"]
        summaries.append(summary)

    return " ".join(summaries)

#set up final detection
def main():
    if uploadfil is None:
        return

    txt = text_extract(uploadfil)

    
    st.text_area("Resume Text", txt, height=300)

    chunks = chunksplit(txt, chunk)

    role_scores = analyze_role_fit(chunks)
    skills = extract_skills(chunks)
    summary = summarize_resume(chunks)

    st.subheader("Resume Summary")
    st.write(summary)

    st.subheader("Role Fit Scores")
    for role, score in sorted(role_scores.items(), key=lambda x: -x[1]):
        st.write(f"{role}: {score}")

    st.subheader("Extracted Skills")
    st.write(skills[:30])

#Run
if __name__ == "__main__":
    main()
