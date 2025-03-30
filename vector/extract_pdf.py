import PyPDF2
   
with open("vector/mes_documents/mspm0l1306.pdf", "rb") as f:
    reader = PyPDF2.PdfReader(f)
    if reader.is_encrypted:
        print("Le PDF est crypté")
    else:
        for page in reader.pages:
            text = page.extract_text()
            print(text)
            if not text:
                print("Aucun texte n'a pu être extrait")