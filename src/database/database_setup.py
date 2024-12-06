import json
import glob
import logging
import sys
import os
from tqdm import tqdm
from dotenv import dotenv_values
from concurrent.futures import ProcessPoolExecutor
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from PyPDF2 import PdfReader

from docling_obj import Docling_Object

import model  
from model import Base, Document
#from papermage.magelib import Document


## Helper functions
def sort_paths_by_file_size(paths):
    """
    Sorts a list of file paths based on the size of the corresponding files.
    
    Parameters:
    - paths (list): List of file paths to sort.
    
    Returns:
    - sorted_paths (list): List of file paths sorted by file size (ascending).
    """
    # Sort the paths by the size of the corresponding file using os.path.getsize
    sorted_paths = sorted(paths, key=lambda path: os.path.getsize(path))
    return sorted_paths



def setup_engine_session(user, password, address, port, db, Base=Base, echo=True):
    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{address}:{port}/{db}", echo=echo)
    session = Session(engine)
    Base.metadata.create_all(engine)
    return session

def positions_from_box(obj):
    return obj.boxes[0].l, obj.boxes[0].t, obj.boxes[0].w, obj.boxes[0].h, obj.boxes[0].page

def extract_data_from_json(path_to_papermage_json, table_root_path):
    """
    Extracts data from the given JSON file and prepares it for the model_class.
    """
    try:
        with open(path_to_papermage_json, "r") as f:
            doc_json = json.load(f)
            current_doc = Document.from_json(doc_json)

        current_doi = path_to_papermage_json.split("/")[-1].replace(".json", "")
        title = current_doc.titles[0].text if current_doc.titles else ""
        authors = [name_.strip() for name_ in current_doc.authors[0].text.split(",")] if current_doc.authors else []

        # Box-like objects
        ir_id = current_doi
        table_paths = sorted(glob.glob(f"{table_root_path}/{ir_id}*"), key=human_sort_key)
        json_extracts = [json.load(open(table_path, "r")) for table_path in table_paths]

        # Table related data
        table_data = []
        for i, table in enumerate(current_doc.tables):
            json_extract = json_extracts[i] if i < len(json_extracts) else None
            l, t, w, h, page = positions_from_box(table)
            t_caps_cleaned = get_cleaned_captions(current_doc)
            ir_tab_id = table_paths[i].split("/")[-1].replace(".json", "") if i < len(table_paths) else ""
            t_caption = get_caption_for_box(box=table.boxes[0], captions=current_doc.captions, caption_ids=t_caps_cleaned["tables"])
            t_caption_refs = get_fulltext_references(current_doc, t_caption)
            table_name = get_table_name_from_caption(t_caption)
            if text_matches_pattern(t_caption, "tables") or text_matches_pattern(table.text, "tables"):
                table_data.append({
                    'text': table.text,
                    'ir_tab_id': ir_tab_id,
                    'ir_id': ir_id,
                    'header': json_extract['header'] if json_extract else None,
                    'content': json_extract['content'] if json_extract else None,
                    'position': (l, t, w, h, page),
                    'caption': t_caption,
                    'table_name': table_name,
                    'references': t_caption_refs,
                })

        # Figure and equation data
        figure_data = [positions_from_box(fig) for fig in current_doc.figures]
        equation_data = [positions_from_box(eq) for eq in current_doc.equations]

        # **Exclude 'current_doc' from the returned data**
        return {
            'current_doi': current_doi,
            'title': title,
            'authors': authors,
            'table_data': table_data,
            'figure_data': figure_data,
            'equation_data': equation_data,
        }
    except Exception as e:
        logging.error(f"Error extracting data from {path_to_papermage_json}: {e}")
        return None

def create_model_objects(data, model_class, session):
    """
    Creates model_class objects using the extracted data and adds them to the session.
    """
    if data is None:
        logging.warning("No data provided to create_model_objects.")
        return

    try:
        p = model_class.Publication()
        current_d = model_class.Document(doi=data['current_doi'], title=data['title'], publication=p)
        current_d.authors = [model_class.Author(name=name_) for name_ in data['authors']]

        # Tables
        tabs = []
        for _, table in enumerate(data['table_data']):
            l, t, w, h, page = table['position']
            t_obj = model_class.Table(
                pm_content=table['text'],
                ir_tab_id=table['ir_tab_id'],
                ir_id=table['ir_id'],
                header=table['header'],
                content=table['content'],
                document_id=data['current_doi'],
                document=current_d,
                caption=table['caption'],
                table_name=table['table_name'],
                position_left=l,
                position_top=t,
                position_page=page,
                width=w,
                height=h,
                references=table['references'],
            )
            tabs.append(t_obj)

        # Figures
        figs = []
        for figure in data['figure_data']:
            l, t, w, h, page = figure
            f = model_class.Figure(
                document_id=data['current_doi'],
                ir_id=data['current_doi'],
                document=current_d,
                position_left=l,
                position_top=t,
                position_page=page,
                width=w,
                height=h
            )
            figs.append(f)

        # Equations
        eqs = []
        for equation in data['equation_data']:
            l, t, w, h, page = equation
            e = model_class.Equation(
                document_id=data['current_doi'],
                document=current_d,
                position_left=l,
                position_top=t,
                position_page=page,
                width=w,
                height=h
            )
            eqs.append(e)

        # Add lists of tables, figures, equations to document
        current_d.tables = tabs
        current_d.figures = figs
        current_d.equations = eqs
        p.documents = [current_d]

        session.add(p)
        for t in tabs:
            session.add(t)
        for f in figs:
            session.add(f)
        for e in eqs:
            session.add(e)
    except Exception as e:
        logging.error(f"Error creating model objects: {e}")

def extract_data_wrapper(json_path):
    return extract_data_from_json(json_path, TABLE_ROOT_PATH)

def parallel_extract_data(json_paths, model, session, num_workers=None, batch_size = 10):
    
    with tqdm(total=len(json_paths), desc="Overall Progress") as pbar:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i in range(0, len(json_paths), batch_size):
                batch_paths = json_paths[i:i + batch_size]

                # Extract data in parallel for the current batch
                with tqdm(total=len(batch_paths), desc="Extracting Data", leave=False) as extract_pbar:
                    extracted_data = []
                    # Using list() to eagerly evaluate executor.map and catch exceptions
                    for data in executor.map(extract_data_wrapper, batch_paths):
                        if data is not None:
                            extracted_data.append(data)
                        extract_pbar.update(1)

                # Prepare session sequentially for the extracted data
                with tqdm(total=len(extracted_data), desc="Preparing Session", leave=False) as prepare_pbar:
                    for data in extracted_data:
                        create_model_objects(data, model, session)
                        prepare_pbar.update(1)
                        pbar.update(1)

                # Commit session after processing the batch
                try:
                    session.commit()
                except Exception as e:
                    logging.error(f"Error committing session: {e}")
                    session.rollback()


def add_pdfs_to_database(folder_path, session, verbose=False, omit_bytes=False, first_n=None):
    # Get the list of PDF files
    if not verbose:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    pdf_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".pdf")][:first_n]
    
    # Iterate through the PDF files with a progress bar
    cnt = 0
    # get all existing DOIs
    avail_dois = set([res[0] for res in session.query(model.Document.doi).all()])
    no_pdf_dois = session.query(Document.doi).filter(Document.pdf.is_(None)).all()

    for filename in tqdm(pdf_files, desc="Processing PDFs"):
        file_path = os.path.join(folder_path, filename)
        
        # Extract metadata (example: using filename as DOI and title)
        doi = os.path.split(file_path)[-1].replace(".pdf", "").replace("$", "/")  # Use filename as DOI

        #title = f"Document for {doi}"  # Example title based on DOI
        # Skip if already in database
        if doi in avail_dois:
            if omit_bytes:
                continue
            else:
                if doi in no_pdf_dois:
                    #update row with pdf
                    doc_2_update = session.get(Document, doi)
                    sanity = is_sane(file_path, verbose=False)
                    if sanity and not omit_bytes:
                        with open(file_path, 'rb') as file:
                            pdf_data = file.read()
                        doc_2_update.pdf = pdf_data
                        doc_2_update.pdf_is_sane = sanity
                        session.commit()

                else:
                    #skip
                    continue
        # Read PDF file as binary
        sanity = is_sane(file_path, verbose=False)
        if sanity and not omit_bytes:
            with open(file_path, 'rb') as file:
                pdf_data = file.read()
        else:
            pdf_data = None
        
        # Create a new Document instance

        p = model.Publication()

        document = Document(
            publication=p,
            doi=doi,
            
            pdf=pdf_data,
            pdf_is_sane=sanity  # Default to False until validation
        )
        
        # Add to session
        session.add(document)
        cnt += 1
        if cnt % 10 == 0:
            session.commit()
    # Commit session
    session.commit()
    print("All PDFs have been added to the database.")


def is_sane(pdf_path: str, size_threshold_kb: int = 15, verbose: bool = False) -> bool:
    """
    Check if a given file is a legitimate PDF.

    Args:
        pdf_path (str): Path to the PDF file.
        size_threshold_kb (int): Minimum file size in KB for the file to be considered valid (default: 15 KB).
        verbose (bool): Whether to print detailed output for debugging (default: False).
    
    Returns:
        bool: True if the file is a legitimate PDF, False otherwise.
    """
    # Check if file exists
    if not os.path.exists(pdf_path):
        if verbose:
            print(f"[ERROR] File does not exist: {pdf_path}")
        return False

    # Check file size (e.g., above size_threshold_kb)
    file_size = os.path.getsize(pdf_path)
    if file_size < size_threshold_kb * 1024:  # Convert KB to bytes
        if verbose:
            print(f"[ERROR] File is too small to be a valid PDF: {file_size} bytes (threshold: {size_threshold_kb * 1024} bytes)")
        return False

    try:
        # Check PDF header
        with open(pdf_path, 'rb') as file:
            header = file.read(5)
            if header != b"%PDF-":
                if verbose:
                    print(f"[ERROR] File does not start with '%PDF-' header: {header}")
                return False
        
        # Check PDF trailer
        with open(pdf_path, 'rb') as file:
            file.seek(-20, os.SEEK_END)  # Read the last 20 bytes
            trailer = file.read()
            if b"%%EOF" not in trailer:
                if verbose:
                    print("[ERROR] File does not contain '%%EOF' trailer.")
                return False

        # Validate PDF structure using PyPDF2
        try:
            reader = PdfReader(pdf_path)
            _ = reader.pages  # Access pages to trigger parsing
        except Exception as e:
            if verbose:
                print(f"[ERROR] PyPDF2 failed to parse the PDF: {e}")
            return False

    except Exception as e:
        if verbose:
            print(f"[ERROR] An error occurred while validating the file: {e}")
        return False

    # All checks passed
    if verbose:
        print(f"[INFO] The file '{pdf_path}' is a legitimate PDF.")
    return True


# Function to save PDFs to local folder
def save_pdfs_to_folder(folder_path, session):
    # Step 1: Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Step 2: Get the list of DOIs with a pdf in the database
    all_dois = session.query(Document.doi).filter(Document.pdf.isnot(None)).all()
    all_dois = {doi[0] for doi in all_dois}  # Extract DOIs from tuples

    # Step 3: Check which DOIs already exist in the folder
    existing_dois = {os.path.splitext(f)[0].replace("$", "/") for f in os.listdir(folder_path) if f.endswith(".pdf")}
 
    missing_dois = all_dois - existing_dois

    if not missing_dois:
        print("All PDFs already exist locally. No downloads needed.")
        return

    print(f"Found {len(missing_dois)} missing PDFs. Downloading...")

    # Step 4: Download missing PDFs
    for doi in tqdm(missing_dois, desc="Downloading PDFs"):
        # Fetch the document from the database
        document = session.query(Document).filter_by(doi=doi).first()
        if document and document.pdf:
            # Save the PDF to the folder
            pdf_path = os.path.join(folder_path, f"{doi.replace('/', '$')}.pdf")
            with open(pdf_path, "wb") as pdf_file:
                pdf_file.write(document.pdf)
        else:
            print(f"[WARNING] No PDF found for DOI: {doi}")

    print("All missing PDFs have been downloaded.")

def add_docling_objects_to_database(docling_out, session, top_k=None):
    cnt = 0
    for file in tqdm(os.listdir(docling_out)[:top_k], desc="Processing JSONs"):
        with open(os.path.join(docling_out, file), "r") as f:
            docling_json = json.load(f)
            docling_obj = Docling_Object(doi=file.replace(".json", "").replace("$", "/"), docling_json=docling_json)
            session.add(docling_obj)
            cnt += 1
        if cnt % 10 == 0:
            session.commit()
    session.commit()


def safe_docling_obj_to_folder(docling_out, session):
    os.makedirs(docling_out, exist_ok=True)

    # Step 2: Get the list of DOIs with a json in the database
    all_dois = session.query(Docling_Object.doi).all()
    all_dois = {doi[0] for doi in all_dois}  # Extract DOIs from tuples
    # Step 3: Check which DOIs already exist in the folder
    existing_dois = {os.path.splitext(f)[0].replace("$", "/") for f in os.listdir(docling_out) if f.endswith(".json")}
    missing_dois = all_dois - existing_dois
    if not missing_dois:
        print("All JSONs already exist locally. No downloads needed.")
        return

    print(f"Found {len(missing_dois)} missing JSONs. Downloading...")

    # Step 4: Download missing JSONs
    for doi in tqdm(missing_dois, desc="Downloading JSONs"):
        # Fetch the object from the database
        json_obj = session.query(Docling_Object).filter_by(doi=doi).first()
        if json_obj:
            # Save the JSON to the folder
            json_file= os.path.join(docling_out, f"{doi.replace('/', '$')}.json")
            with open(json_file, "w") as json_file:
                json.dump(json_obj.docling_json, json_file)
        else:
            print(f"[WARNING] No JSON found for DOI: {doi}")

    print("All missing JSONs have been downloaded.")

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,  # Set to INFO or DEBUG as needed
        format='%(asctime)s %(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler("database_setup.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Disable overly verbose logs if necessary
    logging.disable(logging.FATAL + 1)  # Removed in favor of proper logging configuration

    db_vals = dotenv_values("/workspaces/CORD19_Plus/.env")

    root_parse_path = "/workspaces/CORD19_Plus/data/clean/pub_json2/*.json"
    json_paths = sorted(glob.glob(root_parse_path))

    session = setup_engine_session(
        db_vals['USER'],
        db_vals['PASSWORD'],
        db_vals['ADDRESS'],
        db_vals['PORT'],
        db_vals['DB_FINAL']
    )

    
    # Filter the paths list based on whether the file name (without extension) is in the ID base names
    avail_ids = set([res[0] for res in session.query(model.Document.doi).all()])
    id_base_names = {id.split('_')[0] for id in avail_ids}
    filtered_paths = [path for path in json_paths if path.split('/')[-1].split('.')[0] not in id_base_names]
    print(f"Paths to process: {len(filtered_paths)}")
    json_paths = sort_paths_by_file_size(filtered_paths)

    # Ensure 'model' is defined before passing it
    # Replace with actual model initialization if different
    parallel_extract_data(json_paths, model, session, num_workers=10)


if __name__ == "__main__":
    main()