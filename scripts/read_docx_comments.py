import zipfile
import xml.etree.ElementTree as ET
import sys
import os

def extract_comments(docx_path):
    print(f"Reading comments from {docx_path}")
    try:
        with zipfile.ZipFile(docx_path) as z:
            if 'word/comments.xml' not in z.namelist():
                print("No comments found.")
                return
            xml_content = z.read('word/comments.xml')
            
        tree = ET.fromstring(xml_content)
        # XML namespace for Word
        namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        
        for comment in tree.findall('.//w:comment', namespaces):
            author = comment.get(f"{{{namespaces['w']}}}author", "Unknown")
            texts = [t.text for t in comment.findall('.//w:t', namespaces) if t.text]
            comment_text = "".join(texts)
            print(f"[{author}]: {comment_text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        extract_comments(sys.argv[1])
    else:
        print("Usage: python3 read_docx_comments.py <file.docx>")
