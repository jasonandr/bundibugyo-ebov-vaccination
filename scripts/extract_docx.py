import docx

def extract_docx(filepath):
    print(f"\n--- Extracting: {filepath} ---")
    try:
        doc = docx.Document(filepath)
        text = [p.text for p in doc.paragraphs if p.text.strip()]
        print('\n'.join(text))
    except Exception as e:
        print(f"Error reading docx: {e}")

extract_docx("/Users/jasonandrews/Library/CloudStorage/Dropbox/Isaac-Jason/_Ebola/manuscript/Manuscript_2026_06_27_IB_JA.docx")
