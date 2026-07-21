import docx
import zipfile
import xml.etree.ElementTree as ET

def get_comments(docx_path):
    comments = []
    try:
        zf = zipfile.ZipFile(docx_path)
        if 'word/comments.xml' in zf.namelist():
            xml_content = zf.read('word/comments.xml')
            tree = ET.fromstring(xml_content)
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            for comment in tree.findall('.//w:comment', namespaces):
                author = comment.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}author', 'Unknown')
                texts = [t.text for t in comment.findall('.//w:t', namespaces) if t.text]
                comments.append(f"[{author}]: {''.join(texts)}")
    except Exception as e:
        comments.append(f"Error reading comments: {e}")
    return comments

def extract_docx(filepath):
    print(f"\n--- Extracting: {filepath} ---")
    try:
        doc = docx.Document(filepath)
        text = [p.text for p in doc.paragraphs if p.text.strip()]
        print("TEXT PREVIEW (first 1000 chars):")
        print('\n'.join(text)[:1000])
        print("\nCOMMENTS:")
        comments = get_comments(filepath)
        for c in comments:
            print(c)
    except Exception as e:
        print(f"Error reading docx: {e}")

files = [
    "/Users/jasonandrews/Library/CloudStorage/Dropbox/Isaac-Jason/_Ebola/manuscript/manuscript_2026_06_19_v4_jk_PMK.docx",
    "/Users/jasonandrews/Library/CloudStorage/Dropbox/Isaac-Jason/_Ebola/manuscript/manuscript_2026_06_19_v4_jk_nah.docx",
    "/Users/jasonandrews/Library/CloudStorage/Dropbox/Isaac-Jason/_Ebola/manuscript/manuscript_2026_06_25_v6.docx"
]

for f in files:
    extract_docx(f)

