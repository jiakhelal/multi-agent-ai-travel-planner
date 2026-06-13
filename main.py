import os
import operator
from typing import TypedDict, Annotated
import streamlit as st

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from tools.weather_tool import get_weather

load_dotenv()
print("GROQ KEY =", os.getenv("GROQ_API_KEY"))
print("TAVILY KEY =", os.getenv("TAVILY_API_KEY"))
print("WEATHER KEY =", os.getenv("OPENWEATHER_API_KEY"))


# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=st.secrets["GROQ_API_KEY"]
)

# State
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

    user_query: str

    destination: str
    budget: str
    duration: str

    flight_results: str
    hotel_results: str
    weather_results: str

    budget_analysis: str
    itinerary: str
    final_response: str
    
    
 #destination extractor agent   
def destination_extractor_agent(state: TravelState):

    prompt = f"""
Extract travel information.

Query:
{state['user_query']}

Return exactly:

Destination:
Budget:
Duration:
"""

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    text = response.content

    destination = "Unknown"
    budget = "Unknown"
    duration = "Unknown"

    for line in text.split("\n"):

        if line.lower().startswith("destination"):
            destination = line.split(":", 1)[1].strip()

        elif line.lower().startswith("budget"):
            budget = line.split(":", 1)[1].strip()

        elif line.lower().startswith("duration"):
            duration = line.split(":", 1)[1].strip()

    return {
        "destination": destination,
        "budget": budget,
        "duration": duration,

        "messages": [
            AIMessage(
                content=f"""
Destination: {destination}
Budget: {budget}
Duration: {duration}
"""
            )
        ]
    }




# Flight Agent
def flight_agent(state: TravelState):

    query = state["user_query"]

    raw_results = search_flights(query)

    prompt = f"""
You are a flight analyst.

Search Results:

{raw_results}

Extract:

- Best Airline
- Estimated Round Trip Fare (INR)
- Flight Duration
- Route
- Recommendation

Return clean professional output.

Do NOT include URLs.
Do NOT include snippets.
"""

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    return {
        "flight_results": response.content,
        "messages": [
            AIMessage(content="Flight information collected successfully.")
        ]
    }
    
    
    

# Hotel Agent
def hotel_agent(state: TravelState):

    query = f"""
    Best hotels in {state['destination']}
    suitable for this trip:

    {state['user_query']}
    """

    raw_results = tavily_search(query)

    prompt = f"""
    Convert these hotel search results into a clean report.

    Search Results:
    {raw_results}

    Return ONLY:

    Hotel Name:
    Rating:
    Approx Cost Per Night:
    Why Recommended:

    Give 3-5 hotels.
    """

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    return {
        "hotel_results": response.content,
        "messages": [
            AIMessage(content="Hotel information fetched.")
        ]
    }
    
    
    
#weather agent  
def weather_agent(state: TravelState):

    weather = get_weather(state["destination"])

    prompt = f"""
    Weather Information:

    {weather}

    Create a traveler friendly weather report.

    Include:

    - Temperature
    - Weather Condition
    - Travel Advice
    - What Clothes To Carry
    """

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    return {
        "weather_results": response.content,
        "messages": [
            AIMessage(content="Weather information fetched.")
        ]
    }
    
    
    

# Budget Agent
def budget_agent(state: TravelState):

    prompt = f"""
You are a travel budget analyst.

User Request:
{state['user_query']}

Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Rules:

1. Use ONLY INR (₹)
2. Do NOT use USD
3. Estimate realistic prices
4. Check if trip fits inside budget
5. Give budget breakdown

Return format:

Total Budget: ₹

Flights: ₹
Hotels: ₹
Food: ₹
Transport: ₹
Activities: ₹

Within Budget:
Yes / No

Cost Saving Suggestions:
- ...
- ...
"""

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    return {
        "budget_analysis": response.content,
        "messages": [response]
    }
    
    
    
    

# Itinerary Agent
def itinerary_agent(state: TravelState):
    prompt = f"""
You are an expert travel planner.

Destination:
{state['destination']}

Duration:
{state['duration']}

Budget:
{state['budget_analysis']}

Weather:
{state['weather_results']}

Create a realistic itinerary.

Requirements:

- Day wise plan
- Morning
- Afternoon
- Evening
- Food recommendations
- Transportation recommendations

Do NOT mention URLs.
Do NOT repeat budget details.

Make it concise and professional.
"""

    response = llm.invoke([
        SystemMessage(
            content="You are an expert travel planner."
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response]
    }
    
    
    
    
    
# Final Response Agent
def final_agent(state: TravelState):

    prompt = f"""
Create a professional executive travel report.

Trip Information:

Destination:
{state['destination']}

Duration:
{state['duration']}

Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Weather:
{state['weather_results']}

Budget:
{state['budget_analysis']}

Itinerary:
{state['itinerary']}

Return:

# Executive Summary

# Recommended Flight

# Recommended Hotels

# Weather Insights

# Budget Summary

# Key Highlights

# Travel Tips

IMPORTANT:

Do NOT repeat the entire itinerary.

Do NOT include raw URLs.

Do NOT include Tavily search results.

Keep report concise and professional.
"""

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    return {
        "final_response": response.content,
        "messages": [response]
    }
    
    
    
# ---------------- GRAPH ----------------

graph = StateGraph(TravelState)

graph.add_node("destination_extractor_agent", destination_extractor_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

# ---------------- EDGES ----------------

graph.add_edge(
    START,
    "destination_extractor_agent"
)

# Parallel Agents
graph.add_edge(
    "destination_extractor_agent",
    "flight_agent"
)

graph.add_edge(
    "destination_extractor_agent",
    "hotel_agent"
)

graph.add_edge(
    "destination_extractor_agent",
    "weather_agent"
)

# All three feed Budget Agent
graph.add_edge(
    "flight_agent",
    "budget_agent"
)

graph.add_edge(
    "hotel_agent",
    "budget_agent"
)

graph.add_edge(
    "weather_agent",
    "budget_agent"
)

# Remaining pipeline
graph.add_edge(
    "budget_agent",
    "itinerary_agent"
)

graph.add_edge(
    "itinerary_agent",
    "final_agent"
)

graph.add_edge(
    "final_agent",
    END
)


app=graph.compile()



if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "user_jia"
        }
    }

    user_input = input("Enter travel request: ")

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "destination": "",
            "budget": "",
            "duration": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_analysis": "",
            "itinerary": "",
            "final_response": ""
        },
        config=config
    )

    print("\nFINAL RESPONSE:\n")
    
    print(result["final_response"])
