from langchain_community.document_loaders import (

    PyPDFLoader,
    CSVLoader,
    TextLoader

)

# =========================================================
# LOAD PDF
# =========================================================

def load_pdf(file_path):

    try:

        print(f"\nLoading PDF: {file_path}")

        loader = PyPDFLoader(file_path)

        documents = loader.load()

        print(f"Successfully loaded: {file_path}")

        return documents

    except Exception as e:

        print(f"\nError loading PDF: {file_path}")

        print(f"Reason: {e}")

        return []


# =========================================================
# LOAD CSV
# =========================================================

def load_csv(file_path):

    loader = CSVLoader(file_path=file_path)

    documents = loader.load()

    return documents


# =========================================================
# LOAD TXT
# =========================================================

def load_txt(file_path):

    loader = TextLoader(file_path)

    documents = loader.load()

    return documents