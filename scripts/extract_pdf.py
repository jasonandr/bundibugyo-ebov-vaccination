import sys

def extract_text(pdf_path):
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for i in range(min(3, len(reader.pages))):
                text += reader.pages[i].extract_text()
            print(f"--- {pdf_path} ---")
            print(text[:1500])
    except Exception as e:
        print(f"Error: {e}")
        try:
            import fitz
            doc = fitz.open(pdf_path)
            text = ""
            for i in range(min(3, len(doc))):
                text += doc[i].get_text()
            print(f"--- {pdf_path} ---")
            print(text[:1500])
        except Exception as e2:
             print(f"Error 2: {e2}")

extract_text("Andrews - Lancet - Cholera model 2011.pdf")
extract_text("Liu - mass incarceration and TB in Latin America - Lancet Public Health 2024.pdf")
