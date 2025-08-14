from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import asyncio
from datetime import datetime
import logging
import google.generativeai as genai
from dotenv import load_dotenv
import base64
import io
import json
import re
from PIL import Image as PILImage

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(
    title="Flood Detection API",
    description="Flood risk assessment using Gemini AI",
    version="1.0.0"
)

# CORS middleware - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class CoordinateRequest(BaseModel):
    latitude: float
    longitude: float

class AnalysisResponse(BaseModel):
    success: bool
    risk_level: str
    description: str
    recommendations: list[str]
    elevation: float
    distance_from_water: float
    ai_analysis: str
    message: str

def parse_gemini_response(response_text: str) -> dict:
    """Parse Gemini AI response into structured data"""
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {
            "risk_level": "Medium",
            "description": response_text,
            "recommendations": ["Check local alerts", "Monitor weather"],
            "elevation": 50.0,
            "distance_from_water": 1000.0
        }
    except Exception as e:
        logger.error(f"Error parsing response: {str(e)}")
        return {
            "risk_level": "Medium",
            "description": "Analysis completed",
            "recommendations": ["Stay alert", "Check flood maps"],
            "elevation": 50.0,
            "distance_from_water": 1000.0
        }

@app.get("/")
async def health_check():
    return {"status": "active", "service": "Flood Detection API"}

@app.post("/api/analyze/coordinates")
async def analyze_coordinates(data: CoordinateRequest):
    """Analyze flood risk from coordinates"""
    try:
        logger.info(f"Analyzing coordinates: {data.latitude}, {data.longitude}")
        
        prompt = f"""Analyze flood risk for location:
        Latitude: {data.latitude}
        Longitude: {data.longitude}
        
        Provide JSON with:
        - risk_level (Low/Medium/High/Very High)
        - description
        - 3 recommendations
        - elevation (meters)
        - distance_from_water (meters)"""
        
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            result = parse_gemini_response(response.text)
        except Exception as ai_error:
            logger.error(f"AI error: {str(ai_error)}")
            result = {
                "risk_level": "Medium",
                "description": "Default analysis",
                "recommendations": ["Check alerts", "Prepare supplies"],
                "elevation": 50.0,
                "distance_from_water": 500.0
            }
        
        return {
            "success": True,
            "risk_level": result["risk_level"],
            "description": result["description"],
            "recommendations": result["recommendations"],
            "elevation": result["elevation"],
            "distance_from_water": result["distance_from_water"],
            "ai_analysis": result.get("description", ""),
            "message": "Analysis completed"
        }
        
    except Exception as e:
        logger.error(f"Coordinate error: {str(e)}")
        raise HTTPException(status_code=500, detail="Analysis failed")

@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze flood risk from image"""
    try:
        # Validate image
        if not file.content_type.startswith("image/"):
            raise HTTPException(400, "File must be an image")
        if file.size > 10 * 1024 * 1024:
            raise HTTPException(400, "Max size 10MB")
        
        # Process image
        image = PILImage.open(io.BytesIO(await file.read()))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Analyze with Gemini
        prompt = """Analyze this terrain image for flood risk. 
        Respond with JSON containing:
        - risk_level
        - description  
        - recommendations
        - elevation
        - distance_from_water
        - image_analysis"""
        
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content([prompt, image])
            result = parse_gemini_response(response.text)
        except Exception as ai_error:
            logger.error(f"AI error: {str(ai_error)}")
            result = {
                "risk_level": "Medium",
                "description": "Image analysis failed",
                "recommendations": ["Consult local experts"],
                "elevation": 40.0,
                "distance_from_water": 800.0
            }
        
        return {
            "success": True,
            **result,
            "ai_analysis": result.get("image_analysis", ""),
            "message": "Image analysis completed"
        }
        
    except Exception as e:
        logger.error(f"Image error: {str(e)}")
        raise HTTPException(500, "Image processing failed")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )