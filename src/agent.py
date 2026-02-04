from langchain.agents import Tool, initialize_agent
from langchain_openai import ChatOpenAI


def build_agent(rag_chain):
    tools = [
        Tool(
            name="MedicalKnowledgeBase",
            func=rag_chain.run,
            description="Use for healthcare and medical questions"
        )
    ]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True
    )

    return agent
