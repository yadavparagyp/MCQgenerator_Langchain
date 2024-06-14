import os 
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from src.mcqgenerator.utils import get_table_data,read_file
from src.mcqgenerator.logger import logging
from src.mcqgenerator.mcqgenerator import generate_evaluate_chain
from langchain.callbacks import get_openai_callback

# loading json file
with open("C:\python\Langchain_practice\Response.json","r") as f:
    RESPONSE_JSON=json.load(f)

# creating title for the app
st.title("MCQ Creater Application with LangChain")

with st.form("User Input"):
    uploaded_file=st.file_uploader("Upload Pdf OR Text")
    mcq_count=st.number_input("Number of MCQ",min_value=3,max_value=20)
    subject=st.text_input("Insert Subject",max_chars=20)
    tone=st.text_input("Complexity level of Question",max_chars=20,placeholder='simple')
    button=st.form_submit_button("Create MCQ")

    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("Loading...."):
            try:
                text=read_file(uploaded_file)
                # count tokens and the total cost of API call
                with get_openai_callback() as cb:
                    response=generate_evaluate_chain(
                        {
                        "text": text,
                        "number": mcq_count,
                        "subject":subject,
                        "tone":tone ,
                        "response_json": json.dumps(RESPONSE_JSON)
                        }
                    )
            except Exception as e:
                traceback.print_exception(type(e),e,e.__traceback__)
                st.error('Error')
            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")
                if isinstance(response,dict):
                    # extract the quiz data from response
                    quiz=response.get("quiz",None)
                    if quiz is not None:
                        table_data=get_table_data(quiz)
                        if table_data is not None:
                            df=pd.DataFrame(table_data)
                            df.index=df.index+1
                            st.table(df)
                            # Display the review
                            st.text_area(label="Review",value=response['review'])
                        else:
                            st.error("Error in the table data")
                else:
                    st.write(response)
                
