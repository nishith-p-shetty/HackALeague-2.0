import os
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun, BaseTool
from query_data import query_db
import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from create_database import generate_data_store



# Set gemini pro as llm
llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             verbose = True,
                             temperature = 0.5,
                             google_api_key="AIzaSyD_PWAMO77rDabsHbYZVQQRjZkMcVrmKqM")


#create searches
tool_search = DuckDuckGoSearchRun()

class FetchResearchVD(BaseTool):
    name = "Company Data Fetcher"
    description = "Use this tool when you need to get pre saved data about the company and its products from database"

    def _run(self, query):
        output = query_db(query_text=query, category='research')
        return output

    def _arun(self, quert):
        raise NotImplementedError("This tool does not support async")

# class FetchSalesVD(BaseTool):
#     name = "Vector Database Fetcher"
#     description = "use this tool when you need to get pre saved data from database"

#     def _run(self, query):
#         output = query_db(query_text=query, category='sales')
#         return output

#     def _arun(self, quert):
#         raise NotImplementedError("This tool does not support async")

# class FetchDecisionVD(BaseTool):
#     name ="Vector Database Fetcher"
#     description = "use this tool when you need to get pre saved data from database"

#     def _run(self, query):
#         output = query_db(query_text=query, category='decision')
#         return output

#     def _arun(self, quert):
#         raise NotImplementedError("This tool does not support async")



# Define Agent
Company_Researcher = Agent(
    role='Company Researcher',
    goal="Conduct thorough research on the company's name, brand, products, services, and market position to inform marketing strategies and improve overall business performance.",
    backstory="Assistant designed specifically for corporate research, my expertise lies in gathering and analyzing data related to the company's name, brand, and operations. My goal is to provide accurate and actionable insights that can be used to inform marketing campaigns, product development, and other business initiatives. With a deep understanding of the company's strengths and weaknesses, I am able to help the company make informed decisions that drive growth and success.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[
        FetchResearchVD(),
        tool_search
    ]
)
    

email_author = Agent(
    role='Professional Email Author',
    goal="Create catchy concise compelling email.",
    backstory="Experienced in writing impactful marketing cold email to new customers.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[
        FetchResearchVD()
      ]
)
marketing_strategist = Agent(
    role='Professional Marketing Strategist',
    goal="Lead the team creating marketing campains which lead increased product sale for company.",
    backstory="A seasoned cheif marketing officer with a keen eye for standout marketing content.",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    tools=[
        FetchResearchVD()
      ]
)

content_specialist = Agent(
    role='Content Specialist',
    goal="Critique and refine marketing campains.",
    backstory="A professional copywriter with a wealth of experience in persuasive writing.",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

# Define Task
email_task = Task(
     description='''1. Research about the papaearth company and its products.
    2. Generate two distinct variations of a cold email promoting papaearth products. 
    3. Evaluate the written emails for their effectiveness and engagement.
    4. Scrutinize the emails for grammatical correctness and clarity.
    5. Adjust the emails to align with best practices for cold outreach. Consider the feedback provided to the marketing_strategist.
    6. Revise the emails based on all feedback, creating final versions with include papaearth;s products mentioned products in the final email.''',
    agent=marketing_strategist  # The Marketing Strategist is in charge and can delegate
)
    
ad_task = Task(
     description='''1. Research about the papaearth company and its products.
    2. Generate two distinct variations of a advertising scripts promoting papaearth products. 
    3. Evaluate the written scripts for their effectiveness and engagement.
    4. Scrutinize the scripts for grammatical correctness and clarity.
    5. Consider the feedback provided to the marketing_strategist.
    6. Revise the scripts based on all feedback, creating final versions with include papaearth's products mentioned products in the final scripts.''',
    agent=marketing_strategist  # The Marketing Strategist is in charge and can delegate
)

# Create a Single Crew
email_crew = Crew(
    agents=[email_author, marketing_strategist, content_specialist],
    tasks=[email_task],
    verbose=True,
    process=Process.sequential
)

ad_crew = Crew(
    agents=[Company_Researcher, marketing_strategist, content_specialist],
    tasks=[ad_task],
    verbose=True,
    process=Process.sequential
)



# Execution Flow
#emails_output = agents.kickoff()











def train_data(md_datas):
    #for  doc in web_data_docs:
    #    doc.name.split('.')[0]

    return ""

def get_text_chunks(text):
    
    return ""

def get_vectorstore(text_chunks):
    return ""

def get_conversation_chain(vectorstore):
    return ""

def handle_userinput(user_question):
    # response = st.session_state.conversation({'question': user_question})
    # st.session_state.chat_history = response['chat_history']

    # for i, message in enumerate(st.session_state.chat_history):
    #     if i % 2 == 0:
    #         st.write(user_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)
    #     else:
    #         st.write(bot_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)
    pass

def main():
    load_dotenv()
    st.set_page_config(page_title="Internal Tool", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    
    st.header("Internal Tool :books:")
    
    if st.button("Marketing Email"):
            with st.spinner("Processing"):
                output = email_crew.kickoff()
            st.write(output)
    if st.button("Ad. Script"):
                with st.spinner("Processing"):
                    output = ad_crew.kickoff()
                st.write(output)

    with st.sidebar:
        st.subheader("Your documents")
        md_datas = st.file_uploader("Upload your data's here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                for  doc in md_datas:
                    if doc.name.split('.')[0] == "research":
                        with open("./data/research/" + doc.name, "wb" ) as f:
                            f.write(doc.getbuffer())

                    elif doc.name.split('.')[0] == "sales":
                        with open("./data/sales/" + doc.name, "wb" ) as f:
                            f.write(doc.getbuffer())

                    elif doc.name.split('.')[0] == "decision":
                        with open("./data/decision/" + doc.name, "wb" ) as f:
                            f.write(doc.getbuffer())
                    
                # generate_data_store()
                # raw_text = train_data(md_datas)
                # emails_output = email_crew.kickoff()
                # print(emails_output)
                # # get the text chunks
                # text_chunks = get_text_chunks(raw_text)

                # # create vector store
                # vectorstore = get_vectorstore(text_chunks)

                # # create conversation chain
                # st.session_state.conversation = get_conversation_chain(vectorstore)
                # #st.write(text_chunks)



if __name__ == '__main__':
    # if torch.cuda.is_available():
    #     torch.device("cuda")
    # else:
    #     torch.device("cpu")
    main()