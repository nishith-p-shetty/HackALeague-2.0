import os
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun, BaseTool
from query_data import query_db
import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css




# Set gemini pro as llm
llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             verbose = True,
                             temperature = 0.5,
                             google_api_key="XXXXXX")


#create searches
tool_search = DuckDuckGoSearchRun()

class FetchResearchVD(BaseTool):
    name =  "database fetcher"
    description = "This tool is used to fetch data from  database which contains information"

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
Researcher = Agent(
    role='A resource person who fetches data required from internet',
    goal="Provides a detailed summary of fetched data, highlighting key points",
    backstory="An experienced resource person, who can fetch data from inernet",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[
         tool_search
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




def handle_userinput(user_question):
    task1 = Task(
        description=f''' 
    1. {user_question}
    2. Research about all the question from the internet .
    3. Summarize the info and return it back.
''' ,
    agent=Researcher  # The Marketing Strategist is in charge and can delegate
    )

    # Create a Single Crew
    agents = Crew(
        tasks=[task1],
        agents=[Researcher, Decision_Maker],
        # tasks=[research],
        verbose=True,
        process=Process.sequential,
    )
    output = agents.kickoff()
    return output

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with web data", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    
    st.header("Chat with web data :books:")
    
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
