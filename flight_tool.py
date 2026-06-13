from tools.tavily_tool import tavily_search

def search_flights(query):

    search_query = f"""
    Find flight options for:

    {query}

    Include:
    - airline names
    - approximate ticket prices
    - flight duration
    - best routes
    """

    return tavily_search(search_query)