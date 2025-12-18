from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timezone
import random
from faker import Faker

app = FastAPI(title="Fake Tweet Generator API", version="1.0")

fake = Faker()
Faker.seed(42)

AIRLINES = ['Virgin America', 'United', 'Southwest', 'Delta', 'US Airways', 'American']
SENTIMENTS = ['neutral', 'positive', 'negative']
NEGATIVE_REASONS = [
    None,
    'Bad Flight',
    "Can't Tell",
    'Late Flight',
    'Customer Service Issue',
    'Flight Booking Problems',
    'Lost Luggage',
    'Flight Attendant Complaints',
    'Cancelled Flight',
    'Damaged Luggage',
    'longlines'
]

class Tweet(BaseModel):
    airline_sentiment_confidence: float
    airline: str
    negativereason: Optional[str]
    tweet_created: str 
    text: str

def generate_tweet() -> Tweet:
    airline = random.choice(AIRLINES)
    sentiment = random.choices(
        SENTIMENTS,
        weights=[0.3, 0.25, 0.45],
        k=1
    )[0]

    confidence = round(random.uniform(0.5, 1.0), 3) 
    if sentiment == 'neutral':
        confidence = round(random.uniform(0.3, 0.7), 3)

    negativereason = None
    if sentiment == 'negative':
        negativereason = random.choice(NEGATIVE_REASONS[1:]) 
    elif sentiment == 'neutral':
        negativereason = random.choice(NEGATIVE_REASONS) 

   
    handles = {
        'Virgin America': '@VirginAmerica',
        'United': '@united',
        'Southwest': '@SouthwestAir',
        'Delta': '@Delta',
        'US Airways': '@USAirways',
        'American': '@AmericanAir'
    }
    handle = handles[airline]

    if sentiment == 'positive':
        texts = [
            f"{handle} Great service today — flight was on time and crew was amazing! ✈️👏",
            f"Shoutout to {handle} for upgrading me last minute. You made my day!",
            f"Smooth flight with {handle} — love the new seats and in-flight snacks. 🍪",
            f"{handle} The flight was exceptional, crew very friendly and helpful!",
            f"{handle} Impressed with the punctuality and comfort of this flight!"
        ]
    elif sentiment == 'negative':
        texts = [
            f"{handle} why are your first fares in May over three times more than other carriers when all seats are available to select???",
            f"{handle} flight delayed 4 hours with no updates. Terrible communication. #disappointed",
            f"{handle} lost my luggage AGAIN. This is the third time this year. Unacceptable.",
            f"{handle} customer service hung up on me. What kind of support is that?!",
            f"{handle} 2-hour line at check-in for pre-paid bags. Ridiculous inefficiency.",
        ]
    else: 
        texts = [
            f"{handle} flight was fine. Boarding took a while, but nothing major.",
            f"Average experience with {handle}. On time, but seat was a bit tight.",
            f"Checked in online with {handle}, flight happened. No complaints, no praise.",
            f"{handle} The flight was okay, nothing special but no issues either.",
            f"{handle} Service was average, seats were okay, flight on time."
        ]

    text = random.choice(texts)
    
    if random.random() < 0.3:
        text += " " + fake.sentence(nb_words=6).rstrip(".")

    tweet_created = datetime.now(timezone.utc).isoformat()

    return Tweet(
        airline_sentiment_confidence=confidence,
        airline=airline,
        negativereason=negativereason,
        tweet_created=tweet_created,
        text=text
    )

@app.get("/batch", response_model=List[Tweet])
def get_microbatch(batch_size: int = 10):
    
    if not (1 <= batch_size <= 100):
        batch_size = min(max(batch_size, 1), 100)  
    return [generate_tweet() for _ in range(batch_size)]
