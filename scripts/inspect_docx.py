import sys
import zipfile
import xml.etree.ElementTree as ET

def get_docx_text(path):
    try:
        from docx import Document
        doc = Document(path)
        for i, para in enumerate(doc.paragraphs):
            if para.text.strip():
                print(f"[{i}] {para.text[:100]}...")
    except ImportError:
        print("python-docx not installed, using zipfile")
        document = zipfile.ZipFile(path)
        xml_content = document.read('word/document.xml')
        tree = ET.XML(xml_content)
        WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        PARA = WORD_NAMESPACE + 'p'
        TEXT = WORD_NAMESPACE + 't'
        for i, paragraph in enumerate(tree.iter(PARA)):
            texts = [node.text for node in paragraph.iter(TEXT) if node.text]
            if texts:
                text = "".join(texts)
                print(f"[{i}] {text[:100]}...")

get_docx_text("/Users/jasonandrews/Library/CloudStorage/Dropbox/Isaac-Jason/_Ebola/manuscript/Manuscript_2026_06_27_IB_JA_Rewrite.docx")
