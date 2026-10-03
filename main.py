from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from academic_agent import build_academic_retriever
from fee_agent import build_fee_retriever

load_dotenv()

# Step 1: Initialize retrievers

academic_retriever = build_academic_retriever()
fee_retriever = build_fee_retriever()

# Step 2: Initialize LLM

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# Step 3: Define state

class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


# Step 4: Define nodes

def classifier_node(state: State) -> dict:
    """Classifies the student's query."""

    last_message = state["messages"][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"

        "Use 'academic' for questions about attendance, exams, grading, "
        "credits, promotion, course structure, summer training, "
        "or degree requirements.\n"

        "Use 'fee' for questions about tuition, payment, refund, "
        "late charges, scholarships, or any money-related topic.\n"

        "Use 'general' for greetings, casual talk, or anything "
        "not related to college rules or fees.\n\n"

        f"Query: {last_message}\n\n"

        "Return only one word: academic, fee, or general."
    )

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"

    return {"query_type": category}


def academic_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the academic handbook."""

    query = state["messages"][-1].content

    docs = academic_retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return {"retrieved_context": context}


def fee_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the fee structure."""

    query = state["messages"][-1].content

    docs = fee_retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return {"retrieved_context": context}


def general_node(state: State) -> dict:
    """Handles queries that don't need document retrieval."""

    return {
        "retrieved_context": "NO_RETRIEVAL_NEEDED"
    }


def response_node(state: State) -> dict:
    """Generates the final response."""

    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to "
            f"a {programme} student.\n\n"

            "Answer this question using your general knowledge. "
            "If you are unsure, say so rather than inventing facts.\n\n"

            f"Question: {query}"
        )

    else:
        prompt = (
            f"You are a college assistant helping a "
            f"{programme} student.\n\n"

            "Use the following context from official college "
            "documents to answer accurately.\n"

            "If the context does not contain the answer, "
            "clearly say that the available document does not "
            "provide enough information.\n"

            f"If the context mentions figures for different "
            f"programmes, highlight the one relevant to {programme} "
            "if possible.\n\n"

            f"Context:\n{context}\n\n"

            f"Question: {query}\n\n"

            "Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)

    return {
        "messages": [
            ("ai", response.content.strip())
        ]
    }


# Step 5: Define routing

def route_query(state: State) -> str:
    if state["query_type"] == "academic":
        return "academic_rag"

    elif state["query_type"] == "fee":
        return "fee_rag"

    return "general"


# Step 6: Build graph

graph = StateGraph(State)

graph.add_node("classifier", classifier_node)
graph.add_node("academic_rag", academic_rag_node)
graph.add_node("fee_rag", fee_rag_node)
graph.add_node("general", general_node)
graph.add_node("response", response_node)

graph.add_edge(START, "classifier")

graph.add_conditional_edges(
    "classifier",
    route_query
)

graph.add_edge("academic_rag", "response")
graph.add_edge("fee_rag", "response")
graph.add_edge("general", "response")

graph.add_edge("response", END)

app = graph.compile()


# Step 7: Run the application

print("Welcome to the College Assistant\n")

print("Which programme are you in?")
print("1. BCA")
print("2. BBA")
print("3. B.Com (H)")

choice = input("\nEnter 1, 2 or 3: ")

programme_map = {
    "1": "BCA",
    "2": "BBA",
    "3": "B.Com (H)"
}

student_programme = programme_map.get(choice, "BCA")

print(f"\nGreat! You're set as a {student_programme} student.")
print("Type 'exit' or 'quit' to end the conversation.\n")

while True:
    user_query = input("You: ")

    if user_query.lower() in ["exit", "quit"]:
        break

    result = app.invoke({
        "programme": student_programme,
        "messages": [("human", user_query)]
    })

    print(
        f"\nAssistant: {result['messages'][-1].content}\n"
    )