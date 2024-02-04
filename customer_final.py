import os
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun, BaseTool
from query_data import query_db
import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import ConversationalRetrievalChain

from query_data import query_db




# Set gemini pro as llm
llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             verbose = True,
                             temperature = 0.5,
                             google_api_key="XXXXXXXXX")


#create searches
tool_search = DuckDuckGoSearchRun()

class FetchResearchVD(BaseTool):
    name = "Papaearth database fetcher"
    description = "This tool is used to fetch data from papaearth database which contains information about company and it's products"

    def _run(self, query):
        output = query_db(query_text=query, category='research')
        return output

    def _arun(self, quert):
        raise NotImplementedError("This tool does not support async")

class FetchSalesVD(BaseTool):
    name = "Papaearth's product database fetcher"
    description = "This tool is used to fetch data from papaearth database which contains information about it's products"

    def _run(self, query):
        output = query_db(query_text=query, category='sales')
        return output

    def _arun(self, quert):
        raise NotImplementedError("This tool does not support async")

class FetchDecisionVD(BaseTool):
    name =" Database Fetcher"
    description = "use this tool when you need to get pre saved data from database"

    def _run(self, query):
        output = query_db(query_text=query, category='decision')
        return output

    def _arun(self, quert):
        raise NotImplementedError("This tool does not support async")



# Define Agents
Company_Researcher = Agent(
    role='A resource person who fetches data required from database using tools and also from internet',
    goal="Provides a detailed summary of fetched data, highlighting key points, canalso use internet",
    backstory="An experienced resource person, who can fetch data using their tools to obtain data from database and also from inernet",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[
        FetchResearchVD(), tool_search
      ]
)

# 
Decision_Maker = Agent(
    role='A decision making agent.',
    goal="Based on provided information about two or more products, compares them and returns the best solution to take.",
    backstory="Professional decision maker who weighs pros and cons of each choice and provides the best choice output. ",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    tools=[
        FetchDecisionVD()
      ]
)

Sales_Conversioner = Agent(
    role='Sales Booster',
    goal="Based on provided interests, modify data to boost sales",
    backstory="Assistant designed specifically for sales conversion, my expertise lies in understanding human behavior and motivations. My goal is to help users make informed purchasing decisions by naturally guiding discussions and offering incentives to encourage them to buy. With a deep understanding of the user's goals and pain points, I am able to provide personalized recommendations that align with their interests and needs. Whether it's a promotional discount or a tailored product recommendation, my role is to help users achieve their desired outcome – converting their interest into a sale.",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    tools=[
        FetchSalesVD()
      ]
)

Agent_Selector = Agent(
    role='A facilitator who assigns tasks to agents',
    goal="Based on the question chooses the right agent to perform the task",
    backstory="An well experienced facilitator who assigns tasks to different agents based on question asked to provide best output",
    verbose=True,
    allow_delegation=True,
    llm=llm
)



def handle_userinput(user_question):
    task1 = Task(
        description=f''' 
    1. Research about the company and various products available from the company papaearth.
    2. {user_question}
    3. Summarize the details obtained in consise format.
''' ,
        agent=Company_Researcher  # The Marketing Strategist is in charge and can delegate
    )

    # Create a Single Crew
    agents = Crew(
        tasks=[task1],
        agents=[Agent_Selector, Company_Researcher, Decision_Maker, Sales_Conversioner],
        # tasks=[research],
        verbose=True,
        process=Process.sequential,
    )
    output = agents.kickoff()
    return output


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PapaEarth Data", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header("Chat with PapaEarth :books:")
    
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        with st.spinner("Processing"):
            op = handle_userinput(user_question)
        st.write(op)


   



if __name__ == '__main__':
    # if torch.cuda.is_available():
    #     torch.device("cuda")
    # else:
    #     torch.device("cpu")
    main()
