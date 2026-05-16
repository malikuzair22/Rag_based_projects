from typing import TypedDict, Annotated,Optional
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import  HumanMessage,SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition



load_dotenv()


model = ChatGroq(model="llama-3.3-70b-versatile")

ORDERS_DB = {
    "1234": {"status": "Out for delivery", "item": "Nike Shoes", "date": "May 14, 2026", "carrier": "USPS"},
    "5678": {"status": "Delivered", "item": "Samsung TV", "date": "May 10, 2026", "carrier": "FedEx"},
    "9999": {"status": "Processing", "item": "Apple Watch", "date": "May 18, 2026", "carrier": "UPS"},
}

RETURN_POLICY = {
    "shoes": "30-day return window. Must be unworn and in original packaging.",
    "electronics": "15-day return window. Must be in original packaging with all accessories.",
    "clothing": "45-day return window. Tags must be attached.",
    "default": "30-day return window. Item must be in original condition.",
}

@tool
def order_lookup(order_id : str )->str:
    """Looks up order information based on the order ID."""
    order_info = ORDERS_DB.get(order_id)
    if order_info:
        return f"Order {order_id}: {order_info['item']}- {order_info['status']}- {order_info['date']}- {order_info['carrier']}"
    else:
        return "I don't have any inforamtion about your order"
    
@tool
def return_policy_checker(item_type: str)->str:
    """Looks up return policy based on the product type."""
    item_info = RETURN_POLICY.get(item_type)
    if item_info:
        return f"Policy {item_type}: {item_info}"
    else:
        return RETURN_POLICY["default"]

tools = [order_lookup, return_policy_checker]
model_with_tools = model.bind_tools(tools)
tool_node = ToolNode(tools)


class Customer_agent(TypedDict):
    messages: Annotated[list, add_messages]
    order_no: Optional[str]
    intent : str
    response : str
    active_node: str

def route_after_tools(state: Customer_agent) -> str:
    return state["active_node"]


def intent_node(state: Customer_agent)-> Customer_agent:
    last_message = state['messages'][-1].content
    prompt = f"""Classify this customer message with exactly one word: 'order', 'return', 'faq', or 'escalate'.
    'order' = asking about order status or delivery
    'return' = wants to return or refund something  
    'faq' = general question
    'escalate' = angry, frustrated, or wants a human

    Customer message: {last_message}

    Reply with one word only."""

    result = model.invoke(prompt)
    return { "intent": result.content.strip().lower()}


def route_decision(state: Customer_agent) -> str:
    if state["intent"] == "order":
        return "order_node"
    elif state["intent"] == "return":
        return "return_node"
    elif state["intent"] == "faq":
        return "faq_node"
    else: 
        return "escalate_node"
    

def order_node(state: Customer_agent) -> Customer_agent:
   

    prompt = f"""You are an order status specialist for an ecommerce store.
    Help the customer with the order query.
    """
    full_history = [SystemMessage(content=prompt)] + state["messages"]
    result = model_with_tools.invoke(full_history)
    return {"response": result.content, "messages": [result], "active_node": "order_node"}

def return_node(state: Customer_agent) -> Customer_agent:

    prompt = f"""You are an return/refund specialist for an ecommerce store.
    Help the customer with the return query.
    """
    full_history = [SystemMessage(content=prompt)] + state["messages"]
    result = model_with_tools.invoke(full_history)
    return {"response": result.content, "messages": [result], "active_node":"return_node"}

def faq_node(state: Customer_agent) -> Customer_agent:
    

    prompt = f"""You are an general store FAQ specialist for an ecommerce store.
    Help the customer with the FAQ query.
    """
    full_history = [SystemMessage(content=prompt)] + state["messages"]
    result = model.invoke(full_history)
    return {"response": result.content,"messages": [result]}

def escalate_node(state: Customer_agent) -> Customer_agent:
    prompt = f"""Politely tell the customer a human agent will contact them soon.
    """
    full_history = [SystemMessage(content=prompt)] + state["messages"]
    result = model.invoke(full_history)
    return {"response": result.content, "messages": [result]}



builder = StateGraph(Customer_agent)

builder.add_node("intent_node" , intent_node)
builder.add_node("order_node", order_node)
builder.add_node("return_node", return_node)
builder.add_node("faq_node", faq_node)
builder.add_node("escalate_node", escalate_node)

builder.add_node("tools", tool_node)


builder.set_entry_point("intent_node")

builder.add_conditional_edges("intent_node",route_decision)


builder.add_conditional_edges("order_node", tools_condition)
builder.add_conditional_edges("return_node", tools_condition)
builder.add_conditional_edges("tools", route_after_tools)
builder.add_edge("faq_node", END)
builder.add_edge("escalate_node", END)

checkpointer = MemorySaver()

graph = builder.compile(checkpointer = checkpointer)

config = {"configurable": {"thread_id": "customer-1"}}

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("\n--- Conversation History ---")
        state = graph.get_state(config)
        for msg in state.values["messages"]:
            role = "Human" if isinstance(msg, HumanMessage) else "AI"
            print(f"{role}: {msg.content}\n")

        break
    messages = graph.invoke({"messages": [{"role": "user", "content": user_input}]}, config=config)
    print("Bot: ", messages["messages"][-1].content)


