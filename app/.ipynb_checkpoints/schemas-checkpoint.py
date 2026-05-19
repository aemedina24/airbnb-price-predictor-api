from pydantic import BaseModel

class PredictionInput(BaseModel):
    host_is_superhost: int
    neighbourhood_cleansed: str
    property_type: str
    room_type: str
    accommodates: float
    bathrooms: float
    bedrooms: float
    beds: float
    guests_included: float
    minimum_nights: float
    number_of_reviews: float
    review_scores_rating: float
    instant_bookable: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "host_is_superhost": 0,
                "neighbourhood_cleansed": "copacabana",
                "property_type": "apartment",
                "room_type": "entire homeapt",
                "accommodates": 4.0,
                "bathrooms": 1.0,
                "bedrooms": 1.0,
                "beds": 1.0,
                "guests_included": 1.0,
                "minimum_nights": 1.0,
                "number_of_reviews": 12.0,
                "review_scores_rating": 98.0,
                "instant_bookable": 0
            }
        }

# 🚨 AQUÍ AGREGAMOS EL EJEMPLO DE SALIDA BONITO:
class PredictionOutput(BaseModel):
    estimated_price: float

    class Config:
        json_schema_extra = {
            "example": {
                "estimated_price": 377.25
            }
        }
