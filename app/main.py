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

COUNTER_FILE = "counter.txt"
API_CALL_LIMIT = 700

def get_api_counter():
    if not os.path.exists(COUNTER_FILE):
        return 0
    with open(COUNTER_FILE, "r") as f:
        return int(f.read().strip() or 0)

def increment_api_counter():
    count = get_api_counter() + 1
    with open(COUNTER_FILE, "w") as f:
        f.write(str(count))
    return count

def get_weather(location: str) -> dict:
    """Fetch weather data from OpenWeatherMap API"""
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenWeatherMap API key not set in environment.")
    # 1. Geocode location to get lat/lon
    geocode_url = f"http://api.openweathermap.org/geo/1.0/direct"
    geocode_params = {
        "q": location,
        "limit": 1,
        "appid": api_key
    }
    try:
        geo_response = requests.get(geocode_url, params=geocode_params, timeout=10)
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        if not geo_data:
            raise HTTPException(status_code=404, detail=f"Location '{location}' not found")
        lat = geo_data[0]['lat']
        lon = geo_data[0]['lon']

        # 2. Get weather for lat/lon
        weather_url = f"http://api.openweathermap.org/data/2.5/weather"
        weather_params = {
            "lat": lat,
            "lon": lon,
            "appid": api_key,
            "units": "metric"
        }
        weather_response = requests.get(weather_url, params=weather_params, timeout=10)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        return {
            "temperature": weather_data['main']['temp'],
            "description": weather_data['weather'][0]['description'],
            "humidity": weather_data['main']['humidity'],
            "wind_speed": weather_data['wind']['speed'],
            "precipitation": weather_data.get('rain', {}).get('1h', 0)  # mm in last 1h, if available
        }
    except requests.exceptions.HTTPError as e:
        if weather_response.status_code == 429:
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
    if get_api_counter() >= API_CALL_LIMIT:
        return {"response": "API usage limit reached. Please try again later."}
    try:
        location = extract_location(message.message)
        print(f"Extracted location: {location}")
        weather_data = get_weather(location)
        print(f"Weather data: {weather_data}")
        increment_api_counter()  # Increment after successful weather API call
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