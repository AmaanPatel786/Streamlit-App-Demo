from bs4 import BeautifulSoup
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import json
import os
from langchain_core.prompts import PromptTemplate
import regex as re
from langchain_core.documents import Document
import pandas as pd
import numpy as np


st.set_page_config(page_title="Laptop Shopping AI Assistant", layout='wide')
st.title("Laptop Shopping AI Assistant with Web Scraping")

api_key = os.getenv('OPENAI_API_KEY')

@st.cache_data
def scrape_laptop_data():
  url = "https://webscraper.io/test-sites/e-commerce/static/computers/laptops?page="

  url_lst = []
  for i in range(1,21):
    new_url = url+str(i)
    url_lst.append(new_url)
  print(url_lst)

  res = []

  for url in url_lst:
    response = requests.get(url, timeout=60)
    soup = BeautifulSoup(response.text, "html.parser")
    all_div = soup.find_all('div', class_="col-md-4 col-xl-4 col-lg-4")

    for d in all_div:
      product_name=d.find('a', class_="title").text.strip()
      product_price = d.find('span').text
      description = d.find('p', class_="description card-text").text
      review = d.find('p', class_='review-count float-end').text.strip()

      product_data = {
        'Product Name':product_name,
        'Product Price':product_price,
        'Description':description,
        'Review':review
      }
      res.append(product_data)
  df = pd.DataFrame(res)
  return df

df = scrape_laptop_data()

df['Product Price'] = (
    df['Product Price']
    .str.replace('$', '', regex=False)
    .astype(float)
)

df['Review'] = (
  df['Review']
  .str.replace('reviews','', regex = False)
  .astype(float)
)

def row_to_text(row):
  return f"""
  Product Name: {row['Product Name']}
  Product PriAce: {row['Product Price']}
  Description: {row['Description']}
  Review: {row['Review']}
  """

documents = [
    Document(page_content=text)
    for text in df.apply(row_to_text, axis=1).to_list()
]


embeddings = OpenAIEmbeddings(model='text-embedding-3-small', api_key=api_key)

vectorstores = FAISS.from_documents(documents, embeddings)

llm = ChatOpenAI(
    model='gpt-4o-mini',
    api_key=api_key
)

prompt = PromptTemplate(
    input_variables=['context','question'],
    template="""
    You are a helpful assistant,
    Answer the user query using context below.
    If the answer is not present say:
    "Not available in the provided data"

    context:
    {context}

    Question:
    {question}

    """
)

chain = prompt | llm

def detect_question_type(query):
  q = query.lower()
  if any(x in q for x in ['best','reccomended','suggest']):
    return "RECOMMEND"
  if any(x in q for x in ['list','show','all','below','under','less than']):
    return "FILTER"
  return "FACT"

def extract_price(query):
  match = re.search(r"\$?(\d+)",query)
  return float(match.group(1)) if match else None

def extract_review(query):
    match = re.search(r"(\d+)\s*(?:review|reviews)", query.lower())
    return int(match.group(1)) if match else 0

query = st.text_input("Ask a question about laptops: ")
if query:
  question_type = detect_question_type(query)
  st.caption(f"Detected Question Type is: {question_type}")

  if question_type == "FACT":
    results = vectorstores.similarity_search(query, k=4)
    context = "\n\n".join(r.page_content for r in results)
    answer = chain.invoke({"context":context, "question":query})
    st.success(answer.content)
  elif question_type == "FILTER":
    price_limit = extract_price(query)
    if price_limit is None:
      st.warning("Please set a Price Range.")
    else:
      filtered = df[df['Product Price'] <= price_limit]
      if filtered.empty:
        st.warning("No Laptop Under This Criteria.")
      else:
        docs = filtered.apply(row_to_text, axis=1).to_list()
        context = "\n\n".join(docs[:10])
        answer = chain.invoke({"context":context, "question":"List the laptop names with prices"})
        st.success(answer.content)
  elif question_type == "RECOMMEND":
    price_limit = extract_price(query) or 1000
    review_limit = extract_review(query)

    filtered = df[(df['Product Price'] <= price_limit) & (df['Review']>=review_limit)]
    if filtered.empty:
      st.warning("No suitable laptop found.")
    else:
      docs = filtered.apply(row_to_text, axis=1).to_list()
      context = "\n\n".join(docs[:10])
      answer = chain.invoke({"context":context, "question":"List the laptop names with prices and reviews"})
      st.success(answer.content)

with st.expander("View Your Data"):
  st.dataframe(df)