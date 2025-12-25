import requests
import os
from typing import Optional, Dict
from datetime import datetime

class WeatherService:
    """
    Weather API service using OpenWeatherMap
    Free tier: 1000 calls/day
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        
    def get_weather_by_city(self, city: str, country: str = "BD") -> Dict:
        """
        Get current weather data by city name
        
        Args:
            city: City name (e.g., "Dhaka", "Chittagong")
            country: Country code (default: "BD" for Bangladesh)
        
        Returns:
            Dictionary with weather data
        """
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key not found. Set OPENWEATHER_API_KEY environment variable.")
        
        try:
            params = {
                "q": f"{city},{country}",
                "appid": self.api_key,
                "units": "metric"  # Celsius
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract relevant information
            weather_data = {
                "city": data["name"],
                "country": data["sys"]["country"],
                "temperature": round(data["main"]["temp"], 1),
                "feels_like": round(data["main"]["feels_like"], 1),
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "weather_description": data["weather"][0]["description"],
                "weather_main": data["weather"][0]["main"],
                "wind_speed": data["wind"]["speed"],
                "clouds": data["clouds"]["all"],
                "timestamp": datetime.fromtimestamp(data["dt"]).isoformat(),
                "sunrise": datetime.fromtimestamp(data["sys"]["sunrise"]).strftime("%H:%M:%S"),
                "sunset": datetime.fromtimestamp(data["sys"]["sunset"]).strftime("%H:%M:%S")
            }
            
            # Calculate rainfall (if available)
            # OpenWeatherMap provides rain data in mm for last 1h or 3h
            rainfall = 0
            if "rain" in data:
                rainfall = data["rain"].get("1h", 0) or data["rain"].get("3h", 0)
            
            weather_data["rainfall"] = rainfall
            
            # Calculate soil moisture estimate (simplified)
            # Based on humidity and recent rainfall
            soil_moisture = self._estimate_soil_moisture(
                weather_data["humidity"],
                rainfall,
                weather_data["weather_main"]
            )
            weather_data["estimated_soil_moisture"] = soil_moisture
            
            return weather_data
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching weather data: {str(e)}")
    
    def get_weather_by_coordinates(self, lat: float, lon: float) -> Dict:
        """
        Get current weather data by GPS coordinates
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            Dictionary with weather data
        """
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key not found.")
        
        try:
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric"
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Use same extraction logic as city-based search
            return self._extract_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching weather data: {str(e)}")
    
    def _extract_weather_data(self, data: Dict) -> Dict:
        """Extract and format weather data from API response"""
        weather_data = {
            "city": data.get("name", "Unknown"),
            "country": data["sys"]["country"],
            "temperature": round(data["main"]["temp"], 1),
            "humidity": data["main"]["humidity"],
            "weather_description": data["weather"][0]["description"],
            "timestamp": datetime.fromtimestamp(data["dt"]).isoformat(),
        }
        
        # Rainfall
        rainfall = 0
        if "rain" in data:
            rainfall = data["rain"].get("1h", 0) or data["rain"].get("3h", 0)
        weather_data["rainfall"] = rainfall
        
        # Soil moisture estimate
        weather_data["estimated_soil_moisture"] = self._estimate_soil_moisture(
            weather_data["humidity"],
            rainfall,
            data["weather"][0]["main"]
        )
        
        return weather_data
    
    def _estimate_soil_moisture(self, humidity: float, rainfall: float, weather_condition: str) -> float:
        """
        Estimate soil moisture based on weather conditions
        This is a simplified estimation - real soil moisture needs sensors
        
        Returns: Estimated soil moisture percentage (0-100)
        """
        # Base moisture from humidity
        base_moisture = humidity * 0.6
        
        # Add bonus from rainfall
        if rainfall > 0:
            rain_bonus = min(rainfall * 5, 30)  # Cap at 30%
            base_moisture += rain_bonus
        
        # Adjust based on weather condition
        if weather_condition in ["Rain", "Drizzle", "Thunderstorm"]:
            base_moisture += 10
        elif weather_condition in ["Clear", "Clouds"]:
            base_moisture -= 5
        
        # Keep in realistic range
        return min(max(base_moisture, 10), 95)
    
    def get_forecast(self, city: str, country: str = "BD", days: int = 5) -> Dict:
        """
        Get weather forecast for next few days
        Note: Requires different API endpoint (forecast)
        """
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key not found.")
        
        try:
            forecast_url = "http://api.openweathermap.org/data/2.5/forecast"
            params = {
                "q": f"{city},{country}",
                "appid": self.api_key,
                "units": "metric",
                "cnt": days * 8  # 8 forecasts per day (3-hour intervals)
            }
            
            response = requests.get(forecast_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract daily summaries
            daily_forecasts = []
            for item in data["list"]:
                daily_forecasts.append({
                    "datetime": item["dt_txt"],
                    "temperature": round(item["main"]["temp"], 1),
                    "humidity": item["main"]["humidity"],
                    "weather": item["weather"][0]["description"],
                    "rainfall": item.get("rain", {}).get("3h", 0)
                })
            
            return {
                "city": data["city"]["name"],
                "country": data["city"]["country"],
                "forecasts": daily_forecasts
            }
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching forecast data: {str(e)}")


# Bangladesh major cities coordinates (for quick reference)
BANGLADESH_CITIES = {
    "dhaka": {"lat": 23.8103, "lon": 90.4125},
    "chittagong": {"lat": 22.3569, "lon": 91.7832},
    "khulna": {"lat": 22.8456, "lon": 89.5403},
    "rajshahi": {"lat": 24.3745, "lon": 88.6042},
    "sylhet": {"lat": 24.8949, "lon": 91.8687},
    "barisal": {"lat": 22.7010, "lon": 90.3535},
    "rangpur": {"lat": 25.7439, "lon": 89.2752},
    "mymensingh": {"lat": 24.7471, "lon": 90.4203},
    "comilla": {"lat": 23.4607, "lon": 91.1809},
    "bogra": {"lat": 24.8465, "lon": 89.3770}
}


def get_weather_for_location(location: str, api_key: Optional[str] = None) -> Dict:
    """
    Convenience function to get weather data
    
    Args:
        location: City name or "lat,lon" coordinates
        api_key: OpenWeatherMap API key
    
    Returns:
        Weather data dictionary
    """
    weather_service = WeatherService(api_key)
    
    # Check if location is coordinates (format: "23.8103,90.4125")
    if "," in location and all(part.replace(".", "").replace("-", "").isdigit() 
                               for part in location.split(",")):
        lat, lon = map(float, location.split(","))
        return weather_service.get_weather_by_coordinates(lat, lon)
    else:
        # Assume it's a city name
        return weather_service.get_weather_by_city(location)


if __name__ == "__main__":
    # Test the weather service
    print("Testing Weather Service...")
    print("Note: You need to set OPENWEATHER_API_KEY environment variable")
    
    # Example usage
    try:
        weather = get_weather_for_location("Dhaka")
        print(f"\nWeather in {weather['city']}:")
        print(f"  Temperature: {weather['temperature']}°C")
        print(f"  Humidity: {weather['humidity']}%")
        print(f"  Rainfall: {weather['rainfall']}mm")
        print(f"  Estimated Soil Moisture: {weather['estimated_soil_moisture']:.1f}%")
        print(f"  Description: {weather['weather_description']}")
    except Exception as e:
        print(f"Error: {e}")