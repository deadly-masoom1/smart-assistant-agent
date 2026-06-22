import os
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import numexpr as ne
import uuid
from ddgs import DDGS

llm = ChatGroq(
    api_key= os.environ.get("GROQ_API_KEY") ,
    model = "qwen/qwen3-32b"
)

memory = InMemorySaver()

@tool
def calculator(expression : str) -> str:
    '''calculate the mathematical expression.'''
    result = ne.evaluate(expression)
    return str(result)

calculator_agent = create_agent(llm , [calculator] , checkpointer=memory)

general_agent = create_agent(llm , [] , checkpointer= memory)

@tool
def search(query : str) -> str:
    '''search the internet for real time information , facts , news or any real time data. '''
    try :
        with DDGS() as ddgs:
            result = list(ddgs.text(query , max_results = 3))
            if result :
                output = ""
                for r in result:
                    output += f"{r['title']} , {r['body']}"
                return output
            return "NO result found!"

    except : 
        return "Search Failed!"
    
search_agent = create_agent(llm  , [search] , checkpointer= memory)

def supervisor(question : str , thread_id : str):
    config = {"configurable" : {"thread_id" : thread_id}}

    prompt = f'''You are a supervisor agent. Check the question and assign it to the right agent.
Reply with ONLY one word - no explanation:
"math" - for mathematical calculations
"search" - for real-time information, news, or facts
"general" - for everything else

Question: {question}'''
    
    decision = llm.invoke(prompt).content.strip().lower()

    if "</think>" in decision:
        decision = decision.split("</think>")[-1].strip()

        print(f"DEBUG - Decision: {decision}")

    if "math" in decision :
        result = calculator_agent.invoke(
            {"messages" : [{"role" : "user"  , "content" : question}]},
            config = config
        )
    
    elif "search" in decision :
        result = search_agent.invoke(
            {"messages" : [{"role" : "user" , "content" : question}]} ,
            config = config
        )
    
    else :
        result = general_agent.invoke(
            {"messages" : [{"role" : "user" , "content" : question}]} ,
            config = config
        )

    answer = result["messages"][-1].content.strip()
    if "</think>" in answer:
        answer = answer.split("</think>")[-1].strip()
    
    return answer

thread_id = str(uuid.uuid4())

print(supervisor("what is the current petrol price in Pakistan?" , thread_id))
print("\n\n")

print(supervisor("name of mexico cartel who died recently?" , thread_id))
print("\n\n")

print(supervisor("what is the sum of 5555 and 2?" , thread_id))

print(supervisor("hey.MY name is Abdullah.." , thread_id))
print("\n\n")

print(supervisor("What is my name?" , thread_id))
print("\n\n")


