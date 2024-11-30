import nltk
import re
import streamlit as st
import pickle
import fitz
import docxpy
from sklearn.feature_extraction.text import CountVectorizer

from Resume_Screening import resume_screening_function
nltk.download('stopwords')
nltk.download('punkt')
knn=pickle.load(open('knn.pkl','rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))

#web app

def load_pdf(file):
    text = ""
    pdf_document = fitz.open("pdf", stream=file.read())
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    pdf_document.close()
    return text


def load_doc(file):
    text = ""
    doc = fitz.open("doc", stream=file.read())
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    doc.close()
    return text
def main():
 st.title('Webapp')
 st.sidebar.title('Webapp')
 upload_file = st.file_uploader('Upload Resume File',type=['pdf','txt','docx'])

 if upload_file is not None:
     try:
         resume_bytes=upload_file.read()
         resume_text=resume_bytes.decode('utf-8')
     except UnicodeDecodeError:
         resume_text= resume_bytes.decode('latin-1')
     cleaned_resume = resume_screening_function(resume_text)
     input_features = tfidf.transform([cleaned_resume])
     prediction_id = knn.predict(input_features)[0]

 uploaded_description = st.file_uploader('Upload description File', type=['pdf', 'txt', 'docx'])
 if upload_file is not None:
     try:
         description_bytes=uploaded_description.read()
         description_text=description_bytes.decode('utf-8')
     except UnicodeDecodeError:
         description_text= description_bytes.decode('latin-1')
     text = [resume_text, description_text]

     cv = CountVectorizer()
     count_matrix = cv.fit_transform(text)

     similarity_matrix = (count_matrix)
     print(similarity_matrix)
     match_percentage = similarity_matrix[0][1] * 100
     match_percentage = round(match_percentage, 2)

     category_mapping = {
         15: "Java Developer",
         23: "Testing",
         8: "DevOps Engineer",
         20: "Python Developer",
         24: "Web Designing",
         12: "HR",
         13: "Hadoop",
         3: "Blockchain",
         10: "ETL Developer",
         18: "Operations Manager",
         6: "Data Science",
         22: "Sales",
         16: "Mechanical Engineer",
         1: "Arts",
         7: "Database",
         11: "Electrical Engineering",
         14: "Health and fitness",
         19: "PMO",
         4: "Business Analyst",
         9: "DotNet Developer",
         2: "Automation Testing",
         17: "Network Security Engineer",
         21: "SAP Developer",
         5: "Civil Engineer",
         0: "Advocate",
     }



     category_name = category_mapping.get(prediction_id, "Unknown")
     st.write(prediction_id)
     st.write("Predicted Category:", category_name)
     st.write(f"Your Resume is {match_percentage}% match to the job description!")

if __name__=='__main__':
    main()