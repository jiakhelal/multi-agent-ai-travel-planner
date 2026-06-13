import os 
from tavily import TavilyClient
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
print("TAVILY KEY =", os.getenv("TAVILY_API_KEY"))
client = TavilyClient(
    api_key=st.secrets["TAVILY_API_KEY"]
)

def tavily_search(query):
    try:
        response = client.search(
            query=query,
            max_results=5
        )

        results = []

        for i, r in enumerate(response["results"], 1):
            title = r.get("title", "Unknown")
            url = r.get("url", "Unknown")
            snippet = r.get("content", "").strip()

            if len(snippet) > 300:
                snippet = snippet[:300].rsplit(" ", 1)[0] + "..."

            results.append(
                f"{i}. {title}\nURL: {url}\nSnippet: {snippet}\n"
            )

        return "Here are some search results:\n" + "\n".join(results)

    except Exception as e:
        return f"Tavily Search Error: {str(e)}"
