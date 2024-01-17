import streamlit as st
import os, re, io
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from deep_translator import GoogleTranslator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate

from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.schema.output_parser import StrOutputParser
from PIL import Image
import pytesseract as pt
import fitz
from dotenv import load_dotenv

# Import other required libraries

# Load Private Credentials
load_dotenv()

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust the path accordingly

# Function to convert and extract text from PDF or image
def convert_and_extract_text(uploaded_file, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    image_paths = []  # List to store paths of all processed images

    # Check if the input is a PDF
    if uploaded_file.type == "application/pdf":
        # Read the PDF file from the uploaded file's content
        pdf_content = uploaded_file.read()

        # Open the PDF file using PyMuPDF
        pdf_document = fitz.open("pdf", pdf_content)

        # Iterate over each page in the PDF
        for page_number in range(pdf_document.page_count):
            # Get the page
            page = pdf_document[page_number]

            # Get a pixmap of the page
            pixmap = page.get_pixmap()

            # Create a PIL Image from the pixmap
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

            # Save the image to the output folder
            image_path = os.path.join(output_folder, f"page_{page_number + 1}.png")
            image.save(image_path)

            # Append the image path to the list
            image_paths.append(image_path)

        # Close the PDF document
        pdf_document.close()

    elif uploaded_file.type.startswith("image/"):
        # Input is an image, return its path
        image = Image.open(uploaded_file)
        image_path = os.path.join(output_folder, "uploaded_image.png")
        image.save(image_path)
        image_paths.append(image_path)

    print('pdf2img')
    return image_paths

# Function to extract text from an image
def extract_text(image_path, lang="eng+hin"):
    # Open the image
    img = Image.open(image_path)

    # Experiment with image preprocessing
    # Apply additional preprocessing steps as needed

    # Experiment with different page segmentation modes
    custom_config = r'--psm 6'  # Assume a single uniform block of text

    # Perform OCR with specified language
    text = pt.image_to_string(img, lang=lang, config=custom_config)
    print('pytesseract')
    return text

# Function to translate large text
def translate_large_text(text, chunk_size=500):
    # Break the text into chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # Translate each chunk
    translated_chunks = [GoogleTranslator(source='hi', target='en').translate(chunk) for chunk in chunks]

    # Concatenate the translated chunks
    translated_text = ' '.join(translated_chunks)
    print('hi2en')
    return translated_text

# Function to get summary using LLM
def get_summary(complaint):
    try:
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

        prompt = PromptTemplate.from_template(
            """Provide a concise one-line summary of the main incident in the {complaint}, excluding names, locations, contact information and all irrlevant details and only return the offences committed, for extracting applicable sections and acts.
            """
            )

        output_parser = StrOutputParser()
        chain = prompt | model | output_parser
        summary = chain.invoke({"complaint":complaint})
    
    except:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        model = OpenAI(temperature=0.2)

        prompt = PromptTemplate.from_template(
            """Provide a concise one-line summary of the main incident in the {complaint}, excluding names, locations, contact information and all irrlevant details and only return the offences committed, for extracting applicable sections and acts.
            """
        )

        llm_chain = LLMChain(llm=model, prompt=prompt)
        summarized_dict = llm_chain.invoke({"complaint":complaint})
        summary = summarized_dict['text']

    return summary

# Function to get similar docs
def get_similar_docs(summary, vector_db):
    retriever = vector_db.as_retriever(search_kwargs={"k":4})
    similar_docs = retriever.get_relevant_documents(summary)
    return similar_docs # this will return the top 4 appliable charges on the given complaint
    print('result')

# Function to load CSV data
def load_csv(data):
    loader = CSVLoader(file_path=data, encoding="utf-8", csv_args={'delimiter': ','})
    extracted_data = loader.load()
    return extracted_data

# Function to Create text chunks from extracted_data
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(separators="\n", chunk_size=300)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Function to create vector store
# @st.cache_resource
def vector_db(text_chunks):
    persist_directory = "db/"
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-large',
                                               model_kwargs={"device": "cpu"})
    if os.path.exists(persist_directory) == False:
        vector_db = Chroma.from_documents(documents=text_chunks,
                                          embedding=embeddings, persist_directory=persist_directory)
        vector_db.persist()
    else:
        vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print('vectorDB created')
    return vector_db

# defining final excel file_path
file_path = 'results.xlsx'

# Function to save summary and charges to Excel
# @st.cache_resource
def save_to_excel(summary, applicable_charges):
    # Create DataFrame with new entry
    new_entry = pd.DataFrame({'Summary': [summary], 'Applicable Charges': [applicable_charges]})

    # Check if the file exists
    if os.path.exists(file_path):
        # If file exists, read the existing DataFrame
        existing_df = pd.read_excel(file_path)
        # Append the new entry
        updated_df = existing_df.append(new_entry, ignore_index=True)
    else:
        # If file doesn't exist, use the new entry as the DataFrame
        updated_df = new_entry

    # Save the updated DataFrame to Excel
    updated_df.to_excel(file_path, index=False)

# Function to extract section information using regex
def extract_section_info(charge):
    text = str(charge.page_content)
    pattern = r"Section: Section (\S+) Description: (.+)$"  
    match = re.search(pattern, text)
    if match:
        section_number = match.group(1)
        description = match.group(2)
        return {"Section": section_number, "Description": description}
    else:
        return None

# Streamlit App
def main():
    st.set_page_config(layout='wide')
    st.title("Legal Complaint Analysis App")

    uploaded_file = st.file_uploader("Upload a PDF or image file:", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file:
        image_paths = convert_and_extract_text(uploaded_file, 'output_folder')
        text = extract_text(image_paths[0])  # Assuming only one image is processed
    else:
        text = st.text_area("Enter your complaint:")

    if st.button("Process Complaint"):
        # Translate large text
        st.write(f"Original Text: {text}")  # Debugging info

        complaint = translate_large_text(text)
        st.subheader("Translated Text:")
        st.write(complaint)  # Debugging info

        # Get summary
        summary = get_summary(complaint)
        st.subheader("Summary:")
        st.write(summary)

        # Get similar documents
        vector_database = vector_db(text_split(load_csv('data/ipc-english.csv')))
        charges = get_similar_docs(summary, vector_database)
        # st.write(charges)

        # Extracting section and description from each element in the list
        sections_descriptions = [
            {"Section": (entry.page_content.split("Section ")[1].split("\n")[0]), 
            "Description": entry.page_content.split("Description: ")[1].strip()}
            for entry in charges
]
        # Creating a DataFrame
        df = pd.DataFrame(sections_descriptions)

        # Display the DataFrame
        st.dataframe(df, width=800)

if __name__ == "__main__":
    main()