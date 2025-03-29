import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    
# Use Google's Gemini model instead of Anthropic
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Free tier model
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can i help you research? ")
raw_response = agent_executor.invoke({"query": query, "chat_history": []})

try:
    structured_response = parser.parse(raw_response["output"])
    print(structured_response)
except Exception as e:
    print("Error parsing response", e)
    print("Raw Response: ", raw_response)

# Save the research results to files
if "structured_response" in locals() and structured_response:
    # Create file paths
    base_filename = f"{structured_response.topic[:20].replace(' ', '_')}"
    txt_filepath = f"{base_filename}.txt"
    
    # Save as TXT file
    with open(txt_filepath, "w") as f:
        f.write(f"Research Topic: {structured_response.topic}\n\n")
        f.write(f"Summary:\n{structured_response.summary}\n\n")
        f.write("Sources:\n")
        for source in structured_response.sources:
            f.write(f"- {source}\n")
        f.write("\nTools Used:\n")
        for tool in structured_response.tools_used:
            f.write(f"- {tool}\n")
    
    print(f"\nResearch saved to {txt_filepath}")