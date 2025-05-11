from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import requests
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
client = openai.OpenAI(api_key=openai_api_key
)

class ChatMessage(BaseModel):
    message: str

def get_weather(location: str) -> dict:
    """Fetch weather data from Open-Meteo API"""
    geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
    try:
        geo_response = requests.get(geocoding_url)
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        if not geo_data.get('results'):
            raise HTTPException(status_code=404, detail=f"Location '{location}' not found")
        location_data = geo_data['results'][0]
        latitude = location_data['latitude']
        longitude = location_data['longitude']
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m"
        weather_params = {
            "timeout": 10
        }
        weather_response = requests.get(weather_url, params=weather_params)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        weather_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Foggy", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers", 95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }
        current = weather_data['current']
        return {
            "temperature": current['temperature_2m'],
            "description": weather_codes.get(current['weather_code'], "Unknown"),
            "humidity": current['relative_humidity_2m'],
            "wind_speed": current['wind_speed_10m'],
            "precipitation": current['precipitation']
        }
    except requests.exceptions.HTTPError as e:
        if weather_response.status_code == 429:
            # Return a 429 error to the frontend
            raise HTTPException(status_code=429, detail="Weather API rate limit exceeded. Please try again later.")
        raise HTTPException(status_code=500, detail=f"Weather API error: {str(e)}")
    except Exception as e:
        print("Error in get_weather:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Weather API error: {str(e)}")

def extract_location(message: str) -> str:
    """Use OpenAI to extract location from user message"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract the location name from the user's message. Return only the location name, nothing else."},
                {"role": "user", "content": message}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Error in extract_location:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

@app.post("/chat")
async def chat(message: ChatMessage):
    try:
        location = extract_location(message.message)
        print(f"Extracted location: {location}")
        weather_data = get_weather(location)
        print(f"Weather data: {weather_data}")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful weather assistant. Provide weather information in a friendly, conversational way."},
                {"role": "user", "content": f"Location: {location}\nWeather data: {weather_data}\nUser message: {message.message}"}
            ]
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        print("Error in /chat endpoint:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 