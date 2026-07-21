import zipfile
import xml.etree.ElementTree as ET
import sys

def get_comment_context(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as z:
            if 'word/comments.xml' not in z.namelist() or 'word/document.xml' not in z.namelist():
                print("Missing XML files.")
                return
            comments_xml = z.read('word/comments.xml')
            document_xml = z.read('word/document.xml')
            
        namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        
        # Parse comments
        tree_comments = ET.fromstring(comments_xml)
        comments = {}
        for comment in tree_comments.findall('.//w:comment', namespaces):
            c_id = comment.get(f"{{{namespaces['w']}}}id")
            author = comment.get(f"{{{namespaces['w']}}}author", "Unknown")
            texts = [t.text for t in comment.findall('.//w:t', namespaces) if t.text]
            comments[c_id] = {"author": author, "text": "".join(texts), "highlighted": ""}
            
        # Parse document to find highlighted text
        tree_doc = ET.fromstring(document_xml)
        active_comments = set()
        
        for elem in tree_doc.iter():
            if elem.tag == f"{{{namespaces['w']}}}commentRangeStart":
                c_id = elem.get(f"{{{namespaces['w']}}}id")
                active_comments.add(c_id)
            elif elem.tag == f"{{{namespaces['w']}}}commentRangeEnd":
                c_id = elem.get(f"{{{namespaces['w']}}}id")
                if c_id in active_comments:
                    active_comments.remove(c_id)
            elif elem.tag == f"{{{namespaces['w']}}}t":
                if elem.text:
                    for c_id in active_comments:
                        if c_id in comments:
                            comments[c_id]["highlighted"] += elem.text
                            
        for c_id, data in comments.items():
            print(f"[{data['author']}] Comment: {data['text']}")
            print(f"Highlighted Text: \"{data['highlighted']}\"\n")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_comment_context(sys.argv[1])
