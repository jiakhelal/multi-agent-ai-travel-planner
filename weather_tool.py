import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_weather(city):

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={API_KEY}&units=metric"
    )

    try:

        response = requests.get(url)

        data = response.json()

        if response.status_code != 200:
            return "Weather data unavailable"

        temperature = data["main"]["temp"]

        description = data["weather"][0]["description"]

        humidity = data["main"]["humidity"]

        return f"""
Temperature: {temperature}°C
Weather: {description}
Humidity: {humidity}%
"""

    except Exception as e:

        return f"Weather Error: {str(e)}"