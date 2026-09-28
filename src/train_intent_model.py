"""
TravelMate — Intent Classifier Training Script
==============================================
Trains a machine learning intent classification model using TF-IDF feature
extraction and Logistic Regression. Evaluates test performance, calculates
real metrics (Accuracy, Precision, Recall, F1, Confusion Matrix), and persists
the trained artifacts to models/.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import clean_text

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
os.makedirs(MODEL_DIR, exist_ok=True)

# Supplementary Training Corpus: 26 Travel Intents with 600+ realistic paraphrases
INTENT_DATA = [
    # 1. destination_discovery
    ("Where should I go on vacation?", "destination_discovery"),
    ("Explore new travel destinations in India", "destination_discovery"),
    ("Show me interesting places to visit", "destination_discovery"),
    ("What are some good vacation spots?", "destination_discovery"),
    ("I want to discover somewhere new to visit", "destination_discovery"),
    ("List popular tourist destinations", "destination_discovery"),
    ("Where can I travel this year?", "destination_discovery"),
    ("Find me unique holiday spots", "destination_discovery"),
    ("Give me ideas for my next holiday", "destination_discovery"),
    ("What are the top travel destinations right now?", "destination_discovery"),
    ("Show me hidden gems and offbeat destinations", "destination_discovery"),
    ("Looking for travel inspiration", "destination_discovery"),
    ("Where should I plan my next trip?", "destination_discovery"),
    ("Tell me about popular tourist regions in India", "destination_discovery"),
    ("Discover new hill stations or beaches", "destination_discovery"),
    ("I want to see different places to travel", "destination_discovery"),
    ("What are some wonderful places to see?", "destination_discovery"),
    ("Discover unique travel locations", "destination_discovery"),
    ("Any suggestions for places I haven't seen before?", "destination_discovery"),
    ("Explore beautiful holiday destinations", "destination_discovery"),

    # 2. destination_recommendation
    ("Recommend a nice place to visit for 3 days", "destination_recommendation"),
    ("Suggest places to visit near Mumbai", "destination_recommendation"),
    ("Where should I travel from Delhi for a weekend?", "destination_recommendation"),
    ("Can you suggest a good holiday destination?", "destination_recommendation"),
    ("Recommend some scenic places in Maharashtra", "destination_recommendation"),
    ("Where is a great place to take a break from work?", "destination_recommendation"),
    ("Suggest somewhere to visit with friends", "destination_recommendation"),
    ("I need travel recommendations for south India", "destination_recommendation"),
    ("Recommend destinations with great scenery", "destination_recommendation"),
    ("Where can I go for a short trip?", "destination_recommendation"),
    ("Suggest a good destination for relaxation", "destination_recommendation"),
    ("I want recommendations for places near Pune", "destination_recommendation"),
    ("Give me destination recommendations for next month", "destination_recommendation"),
    ("Suggest some great travel getaways", "destination_recommendation"),
    ("Which tourist spot do you recommend most?", "destination_recommendation"),
    ("I am looking for travel suggestions", "destination_recommendation"),
    ("Suggest places to visit in Goa", "destination_recommendation"),
    ("Suggest somewhere near Mumbai for a weekend", "destination_recommendation"),
    ("Where should we go this weekend?", "destination_recommendation"),
    ("Recommend a destination with good weather", "destination_recommendation"),

    # 3. attraction_recommendation
    ("What are the top attractions in Goa?", "attraction_recommendation"),
    ("What to see in Jaipur?", "attraction_recommendation"),
    ("Must visit places in Mumbai", "attraction_recommendation"),
    ("Show me tourist attractions in Udaipur", "attraction_recommendation"),
    ("What are famous sights in Munnar?", "attraction_recommendation"),
    ("What can I see in Hampi?", "attraction_recommendation"),
    ("Top monuments to visit in Agra", "attraction_recommendation"),
    ("Best sightseeing spots in Manali", "attraction_recommendation"),
    ("What are the main places of interest in Amritsar?", "attraction_recommendation"),
    ("Tell me what to see in Pondicherry", "attraction_recommendation"),
    ("Which attractions shouldn't I miss in Kerala?", "attraction_recommendation"),
    ("What points of interest are in Matheran?", "attraction_recommendation"),
    ("List key sights and forts in Mahabaleshwar", "attraction_recommendation"),
    ("What are the best viewpoints in Lonavala?", "attraction_recommendation"),
    ("Show me popular landmarks in Hyderabad", "attraction_recommendation"),
    ("What should I see in Leh Ladakh?", "attraction_recommendation"),
    ("Tell me about famous temples and ghats in Varanasi", "attraction_recommendation"),
    ("Which beaches should I visit in Gokarna?", "attraction_recommendation"),
    ("Sightseeing spots in Coorg", "attraction_recommendation"),
    ("What attractions are there in Ooty?", "attraction_recommendation"),

    # 4. itinerary_planning
    ("Plan a 3 day trip to Jaipur", "itinerary_planning"),
    ("Make me a 5 day itinerary for Rajasthan", "itinerary_planning"),
    ("How should I plan 4 days in Goa?", "itinerary_planning"),
    ("Create an itinerary for 2 days in Mumbai", "itinerary_planning"),
    ("Give me a day by day plan for Munnar", "itinerary_planning"),
    ("Plan a weekend itinerary for Lonavala", "itinerary_planning"),
    ("How to spend 3 days in Udaipur?", "itinerary_planning"),
    ("Schedule my 4-day vacation in Kerala", "itinerary_planning"),
    ("Plan a 2 day sightseeing trip to Agra and Delhi", "itinerary_planning"),
    ("Make an itinerary for a trip to Hampi", "itinerary_planning"),
    ("Can you create a travel plan for Manali?", "itinerary_planning"),
    ("Plan a 5 day Ladakh tour", "itinerary_planning"),
    ("Generate a 3 day itinerary for Varanasi", "itinerary_planning"),
    ("Detailed day wise schedule for Pondicherry", "itinerary_planning"),
    ("How many days do I need for Amritsar and what schedule?", "itinerary_planning"),
    ("Design an itinerary for Gokarna weekend", "itinerary_planning"),
    ("Make a 4-day travel route for Meghalaya", "itinerary_planning"),
    ("How to organize a 3 day visit to Hyderabad?", "itinerary_planning"),
    ("Create an itinerary for Coorg and Mysore", "itinerary_planning"),
    ("Plan my 2-day Matheran trip", "itinerary_planning"),

    # 5. budget_planning
    ("I have around ₹10000 for a 3 day Goa trip", "budget_planning"),
    ("I have ₹8000 and 3 days from Mumbai", "budget_planning"),
    ("Can I do a trip under ₹5000?", "budget_planning"),
    ("What will a 4 day trip to Jaipur cost?", "budget_planning"),
    ("How much budget do I need for Udaipur?", "budget_planning"),
    ("Calculate the expenses for a weekend in Lonavala", "budget_planning"),
    ("Estimated budget for a week in Ladakh", "budget_planning"),
    ("How expensive is Kerala for 5 days?", "budget_planning"),
    ("I have a budget of 15000 for two people", "budget_planning"),
    ("Plan a trip within ₹7000", "budget_planning"),
    ("What is the cost breakdown for Matheran?", "budget_planning"),
    ("Is ₹12000 enough for 4 days in Manali?", "budget_planning"),
    ("Can I visit Gokarna on a shoestring budget?", "budget_planning"),
    ("How much money should I keep for food and stay in Mumbai?", "budget_planning"),
    ("Budget estimation for traveling to Pondicherry", "budget_planning"),
    ("I have a low budget of ₹6000 where can I go?", "budget_planning"),
    ("Estimate total expenses for Munnar trip", "budget_planning"),
    ("Is Hampi cheap for backpackers?", "budget_planning"),
    ("Cost of traveling to Andaman for 5 days", "budget_planning"),
    ("Break down the budget for a 2 day trip to Mahabaleshwar", "budget_planning"),

    # 6. transportation
    ("How can I travel from Mumbai to Goa?", "transportation"),
    ("What is the best way to reach Lonavala from Mumbai?", "transportation"),
    ("How to get to Matheran from Pune?", "transportation"),
    ("Are there direct flights from Delhi to Jaipur?", "transportation"),
    ("How to reach Gokarna by train?", "transportation"),
    ("What are the transport options from Bangalore to Coorg?", "transportation"),
    ("How can I reach Leh Ladakh by road?", "transportation"),
    ("Best route to drive from Mumbai to Mahabaleshwar", "transportation"),
    ("How to travel from Delhi to Agra?", "transportation"),
    ("Is there a toy train to Ooty?", "transportation"),
    ("How do I reach Munnar from Kochi airport?", "transportation"),
    ("Can I take a ferry from Mumbai to Alibaug?", "transportation"),
    ("How to get to Rishikesh from Delhi by train or bus?", "transportation"),
    ("Distance and travel time between Mumbai and Pune", "transportation"),
    ("How to reach Hampi from Bangalore?", "transportation"),
    ("Best way to commute between Delhi and Manali", "transportation"),
    ("How to reach Amritsar from Delhi?", "transportation"),
    ("Ferry timings and boat rides in Alleppey", "transportation"),
    ("How can I get to Pondicherry from Chennai?", "transportation"),
    ("Transit options from Mumbai to Udaipur", "transportation"),

    # 7. accommodation
    ("Where should I stay in Goa?", "accommodation"),
    ("Recommend budget hotels in Jaipur", "accommodation"),
    ("Best luxury resorts in Udaipur near Lake Pichola", "accommodation"),
    ("Are there good homestays in Coorg?", "accommodation"),
    ("Where to find beachfront shacks in Gokarna?", "accommodation"),
    ("Suggest places to stay in Mumbai near Marine Drive", "accommodation"),
    ("Hostels for backpackers in Rishikesh", "accommodation"),
    ("Best areas to stay in Munnar tea gardens", "accommodation"),
    ("Affordable guesthouses in Hampi", "accommodation"),
    ("Where should I book a hotel in Manali?", "accommodation"),
    ("Houseboat stays in Alleppey recommendations", "accommodation"),
    ("Suggest family-friendly hotels in Mahabaleshwar", "accommodation"),
    ("Good heritage havelis to stay in Jaisalmer", "accommodation"),
    ("Where to stay in Pondicherry French Quarter?", "accommodation"),
    ("Resorts in Lonavala with scenic view", "accommodation"),
    ("Hotels near Golden Temple in Amritsar", "accommodation"),
    ("Safe accommodations in Varanasi near the ghats", "accommodation"),
    ("Where to stay in Leh city?", "accommodation"),
    ("Cheap accommodations in Matheran", "accommodation"),
    ("Best boutique stays in Kochi", "accommodation"),

    # 8. food_recommendation
    ("What food should I try in Hyderabad?", "food_recommendation"),
    ("What should I eat in Maharashtra?", "food_recommendation"),
    ("I'm vegetarian. What food should I try in Pune?", "food_recommendation"),
    ("What local food is famous in Jaipur?", "food_recommendation"),
    ("Must-try street food in Mumbai", "food_recommendation"),
    ("Famous dishes and seafood in Goa", "food_recommendation"),
    ("What cuisine is famous in Kerala?", "food_recommendation"),
    ("Authentic food to try in Amritsar", "food_recommendation"),
    ("Tell me about Banarasi street food and sweets", "food_recommendation"),
    ("What should I eat in Udaipur?", "food_recommendation"),
    ("Famous sweets and fudge in Lonavala", "food_recommendation"),
    ("Where to eat authentic Dal Baati Churma?", "food_recommendation"),
    ("Best places to eat Hyderabadi Dum Biryani", "food_recommendation"),
    ("Vegetarian food specialties in Rajasthan", "food_recommendation"),
    ("What is Pithla Bhakri and where to eat it?", "food_recommendation"),
    ("Strawberries and cream in Mahabaleshwar", "food_recommendation"),
    ("Culinary specialties of Pondicherry French cafes", "food_recommendation"),
    ("Local food to taste in Manali and Himachal", "food_recommendation"),
    ("What breakfast items should I try in South India?", "food_recommendation"),
    ("Best tea and snacks in Darjeeling", "food_recommendation"),

    # 9. activity_recommendation
    ("What activities can I do in Goa?", "activity_recommendation"),
    ("Fun things to do in Manali", "activity_recommendation"),
    ("Adventure sports in Rishikesh", "activity_recommendation"),
    ("What to do in Udaipur for fun?", "activity_recommendation"),
    ("Activities for tourists in Mumbai", "activity_recommendation"),
    ("Things to do in Lonavala on a weekend", "activity_recommendation"),
    ("What can I do in Munnar besides visiting tea gardens?", "activity_recommendation"),
    ("Water sports and scuba diving in Andaman", "activity_recommendation"),
    ("What activities are available in Matheran?", "activity_recommendation"),
    ("Things to do in Jaipur in the evening", "activity_recommendation"),
    ("Trekking and camping activities in Gokarna", "activity_recommendation"),
    ("Boat rides and activities in Alleppey backwaters", "activity_recommendation"),
    ("What can tourists do in Hampi?", "activity_recommendation"),
    ("Fun activities in Pondicherry", "activity_recommendation"),
    ("What to do in Jaisalmer desert?", "activity_recommendation"),
    ("Things to do in Mahabaleshwar", "activity_recommendation"),
    ("Activities to do in Coorg coffee estates", "activity_recommendation"),
    ("What can I experience in Varanasi?", "activity_recommendation"),
    ("Sightseeing and things to do in Leh", "activity_recommendation"),
    ("Activities in Pune for travelers", "activity_recommendation"),

    # 10. best_time_to_visit
    ("When is the best time to visit Goa?", "best_time_to_visit"),
    ("Best time to go to Rajasthan", "best_time_to_visit"),
    ("When should I travel to Leh Ladakh?", "best_time_to_visit"),
    ("Which month is best for Manali snow?", "best_time_to_visit"),
    ("When is the ideal season to visit Munnar?", "best_time_to_visit"),
    ("What is the best time of year to visit Varanasi?", "best_time_to_visit"),
    ("When should I plan a trip to Kerala?", "best_time_to_visit"),
    ("Best months to explore Hampi without excessive heat", "best_time_to_visit"),
    ("When is the ideal time to travel to Kashmir?", "best_time_to_visit"),
    ("What is the peak tourist season in Jaipur?", "best_time_to_visit"),
    ("Best time of year to visit Andaman Islands", "best_time_to_visit"),
    ("Is October a good time for Darjeeling?", "best_time_to_visit"),
    ("When should I visit Ooty?", "best_time_to_visit"),
    ("Best months to visit Rishikesh for river rafting", "best_time_to_visit"),
    ("When does it snow in Shimla?", "best_time_to_visit"),
    ("What is the best time to visit Mahabaleshwar?", "best_time_to_visit"),
    ("When is the best season to explore Pondicherry?", "best_time_to_visit"),
    ("Best time of the year for Gokarna beaches", "best_time_to_visit"),
    ("Which month is ideal for Udaipur lakes?", "best_time_to_visit"),
    ("When to visit Amritsar Golden Temple?", "best_time_to_visit"),

    # 11. seasonal_travel
    ("What are some underrated places to visit during monsoon?", "seasonal_travel"),
    ("Best places to visit in summer in India", "seasonal_travel"),
    ("Where to go during winter vacation?", "seasonal_travel"),
    ("Top destinations for monsoon travel in Maharashtra", "seasonal_travel"),
    ("Where can I travel in July and August?", "seasonal_travel"),
    ("Best winter getaways with warm weather", "seasonal_travel"),
    ("Where should I go to escape the summer heat?", "seasonal_travel"),
    ("Great autumn travel destinations in October", "seasonal_travel"),
    ("Where to travel during spring season?", "seasonal_travel"),
    ("Rainy season holiday destinations", "seasonal_travel"),
    ("Cool hill stations to visit in May and June", "seasonal_travel"),
    ("Places to visit during Diwali holidays", "seasonal_travel"),
    ("Where is it pleasant to travel in December?", "seasonal_travel"),
    ("Monsoon trips near Mumbai and Pune", "seasonal_travel"),
    ("Snowy winter destinations in North India", "seasonal_travel"),
    ("Destinations good for travel in January", "seasonal_travel"),
    ("Best places to visit during Christmas vacation", "seasonal_travel"),
    ("Summer vacation destinations for family", "seasonal_travel"),
    ("Where can I see lush waterfalls during rains?", "seasonal_travel"),
    ("Offbeat monsoon getaways", "seasonal_travel"),

    # 12. family_travel
    ("I'm travelling with my parents and they can't walk too much. Where should we go?", "family_travel"),
    ("I am travelling with my parents", "family_travel"),
    ("Best holiday destinations for a family with kids", "family_travel"),
    ("Where should I travel with elderly parents?", "family_travel"),
    ("Family vacation ideas in South India", "family_travel"),
    ("Suggest places with gentle walking and easy accessibility for seniors", "family_travel"),
    ("Safe and comfortable destinations for family trip", "family_travel"),
    ("Vacation spots suitable for grandparents and children", "family_travel"),
    ("Where can a family relax together for 4 days?", "family_travel"),
    ("Family-friendly destinations in Maharashtra", "family_travel"),
    ("Where to go with parents who prefer temples and calm places?", "family_travel"),
    ("Suggest scenic places where lots of climbing is not required", "family_travel"),
    ("Kid friendly vacation spots in India", "family_travel"),
    ("Comfortable travel recommendations for older travelers", "family_travel"),
    ("Where to travel with parents in winter?", "family_travel"),
    ("Family tour packages ideas in Rajasthan", "family_travel"),
    ("Destinations with low walking requirement for seniors", "family_travel"),
    ("Peaceful family vacation without hectic travel", "family_travel"),
    ("Best family retreats with minimal trekking", "family_travel"),
    ("Where should I take my parents for their anniversary?", "family_travel"),

    # 13. solo_travel
    ("Best destinations for solo travel in India", "solo_travel"),
    ("I am planning a solo trip where should I go?", "solo_travel"),
    ("Safe places for female solo travelers", "solo_travel"),
    ("Solo backpacking destinations on a budget", "solo_travel"),
    ("Where can I travel alone peacefully?", "solo_travel"),
    ("Suggest places for first-time solo traveler", "solo_travel"),
    ("Solo trip to Himachal Pradesh recommendations", "solo_travel"),
    ("Is Gokarna good for solo travelers?", "solo_travel"),
    ("Where can I meet other backpackers on a solo trip?", "solo_travel"),
    ("Solo traveler friendly hostels and towns", "solo_travel"),
    ("I want to travel alone to introspect and write", "solo_travel"),
    ("Best places in South India for solo travelers", "solo_travel"),
    ("Solo trip ideas for a 3-day weekend", "solo_travel"),
    ("Safe destinations for woman traveling alone", "solo_travel"),
    ("Solo travel in Rishikesh or Dharamshala", "solo_travel"),
    ("I want to do a solo road trip", "solo_travel"),
    ("Top places for a solo nature retreat", "solo_travel"),
    ("Where to go solo from Bangalore?", "solo_travel"),
    ("Solo backpacking across Rajasthan", "solo_travel"),
    ("Planning my first trip by myself", "solo_travel"),

    # 14. romantic_travel
    ("Plan something romantic but affordable", "romantic_travel"),
    ("Best honeymoon destinations in India", "romantic_travel"),
    ("Romantic weekend getaways for couples", "romantic_travel"),
    ("Suggest quiet romantic places near Mumbai", "romantic_travel"),
    ("Where should I go for our first anniversary?", "romantic_travel"),
    ("Cozy hill stations for couples", "romantic_travel"),
    ("Romantic beach destinations for a couple", "romantic_travel"),
    ("Intimate holiday spots with candlelight dining", "romantic_travel"),
    ("Is Udaipur good for a romantic trip?", "romantic_travel"),
    ("Suggest a scenic romantic resort getaway", "romantic_travel"),
    ("Places to visit with my wife for vacation", "romantic_travel"),
    ("Romantic places in Kerala with houseboats", "romantic_travel"),
    ("Where can couples enjoy peaceful sunsets?", "romantic_travel"),
    ("Honeymoon spots in North East India", "romantic_travel"),
    ("Romantic travel ideas on a modest budget", "romantic_travel"),
    ("Fairy-tale romantic destinations in India", "romantic_travel"),
    ("Suggest a special trip for two", "romantic_travel"),
    ("Private and uncrowded romantic retreats", "romantic_travel"),
    ("Scenic spots to propose or celebrate romance", "romantic_travel"),
    ("Romantic lakeside destinations", "romantic_travel"),

    # 15. adventure_travel
    ("Where can I find thrilling adventure sports?", "adventure_travel"),
    ("I want adventure activities near Pune", "adventure_travel"),
    ("Best places for river rafting and bungee jumping", "adventure_travel"),
    ("Where to go for scuba diving in India?", "adventure_travel"),
    ("Trekking and paragliding destinations in Himachal", "adventure_travel"),
    ("Adrenaline-pumping trips for young people", "adventure_travel"),
    ("High-altitude adventure tours in Ladakh", "adventure_travel"),
    ("Thrilling bike trip routes in the mountains", "adventure_travel"),
    ("Where can I do rock climbing and bouldering?", "adventure_travel"),
    ("Extreme sports vacation spots", "adventure_travel"),
    ("Water sports and parasailing destinations", "adventure_travel"),
    ("Best treks in the Western Ghats", "adventure_travel"),
    ("Adventure weekend getaways from Mumbai", "adventure_travel"),
    ("Skiing and snowboarding places in India", "adventure_travel"),
    ("Where to go for wild jungle safaris?", "adventure_travel"),
    ("White water rafting destinations", "adventure_travel"),
    ("Cliff jumping and cave exploration spots", "adventure_travel"),
    ("Exciting adventure trip with college friends", "adventure_travel"),
    ("Off-roading and jeep safari destinations", "adventure_travel"),
    ("Thrilling travel experiences in India", "adventure_travel"),

    # 16. nature_travel
    ("Suggest peaceful nature destinations", "nature_travel"),
    ("I love photography and waterfalls. Suggest somewhere near Mumbai", "nature_travel"),
    ("Where can I find green forests and quiet valleys?", "nature_travel"),
    ("Best wildlife sanctuaries and birdwatching spots", "nature_travel"),
    ("I want to disconnect from everything and be in nature", "nature_travel"),
    ("Suggest scenic hill stations surrounded by tea gardens", "nature_travel"),
    ("Nature photography holiday destinations", "nature_travel"),
    ("Where can I see scenic lakes and mountains?", "nature_travel"),
    ("Quiet nature retreat away from city pollution", "nature_travel"),
    ("Beautiful landscapes and waterfalls in Western Ghats", "nature_travel"),
    ("I love mountains and mist, suggest places to visit", "nature_travel"),
    ("Lush green destinations for nature lovers", "nature_travel"),
    ("Peaceful lakeside getaways", "nature_travel"),
    ("Where can I see national parks and tigers?", "nature_travel"),
    ("Scenic viewpoints and valleys in South India", "nature_travel"),
    ("Nature trails and botanical gardens", "nature_travel"),
    ("Unspoiled pristine nature spots", "nature_travel"),
    ("Where to see natural beauty in Northeast India?", "nature_travel"),
    ("Quiet mountain villages with great nature", "nature_travel"),
    ("Where to travel for fresh air and greenery?", "nature_travel"),

    # 17. historical_travel
    ("Best historical places to visit in India", "historical_travel"),
    ("I love ancient architecture and forts", "historical_travel"),
    ("Show me UNESCO World Heritage sites to visit", "historical_travel"),
    ("Places rich in Maratha history and hill forts", "historical_travel"),
    ("Historical palaces and monuments of Rajasthan", "historical_travel"),
    ("Where can I see ancient rock-cut cave temples?", "historical_travel"),
    ("Tell me about historic cities with rich heritage", "historical_travel"),
    ("Destinations for history buffs and architecture lovers", "historical_travel"),
    ("Explore ancient ruins of Vijayanagara empire in Hampi", "historical_travel"),
    ("Mughal heritage and architecture tour in India", "historical_travel"),
    ("Colonial French and Portuguese historic towns", "historical_travel"),
    ("Where can I explore old museums and palaces?", "historical_travel"),
    ("Heritage walk destinations in India", "historical_travel"),
    ("Historical significance of Ajanta and Ellora", "historical_travel"),
    ("Ancient cities with thousands of years of history", "historical_travel"),
    ("Historic landmarks to see in Delhi and Agra", "historical_travel"),
    ("Forts to visit around Pune and Maharashtra", "historical_travel"),
    ("Royal monuments and living forts in Jaisalmer", "historical_travel"),
    ("Cultural and architectural heritage trips", "historical_travel"),
    ("History tours in South India", "historical_travel"),

    # 18. religious_travel
    ("Spiritual destinations for pilgrimage in India", "religious_travel"),
    ("Where should I go for temple darshan with family?", "religious_travel"),
    ("Sacred ghats and Ganga aarti destinations", "religious_travel"),
    ("Famous Jyotirlinga temples in Maharashtra", "religious_travel"),
    ("Tell me about holy pilgrimage to Shirdi Sai Baba", "religious_travel"),
    ("Spiritual ashrams and meditation centers in Rishikesh", "religious_travel"),
    ("Visiting the holy Golden Temple in Amritsar", "religious_travel"),
    ("Sacred rituals and temples of Varanasi", "religious_travel"),
    ("Spiritual peace and Buddhist monasteries in Ladakh", "religious_travel"),
    ("Famous temples in South India to visit", "religious_travel"),
    ("Pilgrimage circuits and holy cities", "religious_travel"),
    ("Where to go for spiritual healing and meditation?", "religious_travel"),
    ("Spiritual trip for grandparents", "religious_travel"),
    ("Sacred churches of Old Goa", "religious_travel"),
    ("Where to find serenity and prayer in the Himalayas?", "religious_travel"),
    ("Religious tours across Uttar Pradesh and Uttarakhand", "religious_travel"),
    ("Atmalinga temple at Gokarna significance", "religious_travel"),
    ("Tibetan Buddhist monasteries in Dharamshala", "religious_travel"),
    ("Spiritual retreat to renew energy", "religious_travel"),
    ("Pilgrim places easily accessible by senior citizens", "religious_travel"),

    # 19. weekend_trip
    ("Quick weekend trip ideas from Mumbai", "weekend_trip"),
    ("Where can I go for a 2 day weekend from Pune?", "weekend_trip"),
    ("Weekend getaways near Delhi under 4 hours", "weekend_trip"),
    ("Short 2-day escape from city rush", "weekend_trip"),
    ("Where should I go this Saturday and Sunday?", "weekend_trip"),
    ("Weekend road trips from Bangalore", "weekend_trip"),
    ("Best spots for a quick 48 hour vacation", "weekend_trip"),
    ("Easy weekend trip destinations", "weekend_trip"),
    ("Quick hill station break for the weekend", "weekend_trip"),
    ("Weekend getaway near Hyderabad", "weekend_trip"),
    ("Where to drive for a weekend near Mumbai?", "weekend_trip"),
    ("Short weekend beach trip", "weekend_trip"),
    ("Relaxing 2-day trip ideas", "weekend_trip"),
    ("Where to spend the coming long weekend?", "weekend_trip"),
    ("Quick nature weekend trip", "weekend_trip"),
    ("Weekend camping spots near Pune", "weekend_trip"),
    ("2 day itinerary for a quick break", "weekend_trip"),
    ("Weekend getaways with direct train connectivity", "weekend_trip"),
    ("Where can I travel for only two days?", "weekend_trip"),
    ("Short weekend vacation planning", "weekend_trip"),

    # 20. destination_comparison
    ("Which is better for a weekend, Lonavala or Matheran?", "destination_comparison"),
    ("Compare Goa and Gokarna for a budget trip", "destination_comparison"),
    ("Manali vs Shimla which is better for snowfall?", "destination_comparison"),
    ("Ooty or Munnar which should I choose?", "destination_comparison"),
    ("Which is better between Jaipur and Udaipur?", "destination_comparison"),
    ("Compare Rishikesh and Haridwar", "destination_comparison"),
    ("Is Coorg better than Wayanad for nature?", "destination_comparison"),
    ("Compare Mahabaleshwar and Lonavala for a family trip", "destination_comparison"),
    ("Which destination is cheaper, Pondicherry or Goa?", "destination_comparison"),
    ("Darjeeling vs Gangtok for mountain views", "destination_comparison"),
    ("Should I go to Alibaug or Kashid for beach?", "destination_comparison"),
    ("Compare Hampi and Badami for heritage lovers", "destination_comparison"),
    ("Which is less crowded, Matheran or Mahabaleshwar?", "destination_comparison"),
    ("Compare North Goa and South Goa", "destination_comparison"),
    ("Which is better for parents, Ooty or Kodaikanal?", "destination_comparison"),
    ("Leh Ladakh vs Spiti Valley comparison", "destination_comparison"),
    ("Which city has better food, Mumbai or Hyderabad?", "destination_comparison"),
    ("Compare budget between Agra and Jaipur", "destination_comparison"),
    ("Which place is more peaceful, Gokarna or Goa?", "destination_comparison"),
    ("Differences between visiting Shimla and Manali", "destination_comparison"),

    # 21. packing_advice
    ("I'm going to Rajasthan next month. What should I pack?", "packing_advice"),
    ("What should I pack for Ladakh?", "packing_advice"),
    ("What should I carry for a monsoon trip?", "packing_advice"),
    ("Packing checklist for a beach vacation in Goa", "packing_advice"),
    ("What clothes to wear in Manali in winter?", "packing_advice"),
    ("What should I pack for high altitude mountain trek?", "packing_advice"),
    ("Essential items to pack for traveling with parents", "packing_advice"),
    ("Packing tips for desert safari in Jaisalmer", "packing_advice"),
    ("What shoes are recommended for trekking in Western Ghats?", "packing_advice"),
    ("What should I carry for temple visits and ghats in Varanasi?", "packing_advice"),
    ("Clothing advice for women traveling to temple towns", "packing_advice"),
    ("What medicines and first aid to carry while traveling?", "packing_advice"),
    ("Winter clothes packing list for Shimla and Kufri", "packing_advice"),
    ("What electronics and accessories should I take on a trip?", "packing_advice"),
    ("Packing light for a 3 day solo trip", "packing_advice"),
    ("Things to carry for a water sports trip in Andaman", "packing_advice"),
    ("What to pack for Kerala backwaters and monsoon?", "packing_advice"),
    ("Rain gear essentials to bring for monsoon travel", "packing_advice"),
    ("What to wear during wildlife safari?", "packing_advice"),
    ("Checklist of essential items for domestic travel in India", "packing_advice"),

    # 22. travel_tips
    ("Travel tips for visiting India", "travel_tips"),
    ("Useful advice for first-time visitors to Goa", "travel_tips"),
    ("Tips for visiting Taj Mahal without long queues", "travel_tips"),
    ("How to avoid tourist scams in big cities?", "travel_tips"),
    ("Tips for comfortable train travel on Indian railways", "travel_tips"),
    ("General guidelines for bargaining in local bazaars", "travel_tips"),
    ("How to acclimatize properly in Leh Ladakh?", "travel_tips"),
    ("Photography tips and rules at historical monuments", "travel_tips"),
    ("Tips for traveling during peak tourist season", "travel_tips"),
    ("How to save money while traveling across India?", "travel_tips"),
    ("Advice for renting scooters in Goa or Gokarna", "travel_tips"),
    ("Tips for hiring reliable local guides", "travel_tips"),
    ("Important advice for road trips in the Western Ghats", "travel_tips"),
    ("How to dress respectfully at sacred religious shrines?", "travel_tips"),
    ("Tips for vegetarian travelers dining in unfamiliar towns", "travel_tips"),
    ("Advice on staying hydrated and healthy while traveling", "travel_tips"),
    ("Best travel apps to use for navigation and booking", "travel_tips"),
    ("Tips for attending the Ganga Aarti in Varanasi", "travel_tips"),
    ("How to handle currency exchange and digital payments in India?", "travel_tips"),
    ("Practical advice for planning a smooth vacation", "travel_tips"),

    # 23. safety_advice
    ("What precautions should I take while travelling solo?", "safety_advice"),
    ("Is it safe to visit the Western Ghats in heavy rain?", "safety_advice"),
    ("Safety guidelines for female solo backpackers", "safety_advice"),
    ("What precautions are needed for mountain road trips?", "safety_advice"),
    ("Is street food in Mumbai safe to eat?", "safety_advice"),
    ("Safety tips for swimming in the Arabian Sea at beaches", "safety_advice"),
    ("Emergency numbers and contacts to keep while traveling", "safety_advice"),
    ("How safe is nighttime travel by public bus?", "safety_advice"),
    ("What to do in case of altitude sickness in Ladakh?", "safety_advice"),
    ("Health precautions against waterborne diseases and malaria", "safety_advice"),
    ("Are water sports in Goa certified and safe?", "safety_advice"),
    ("Precautions to take when hiking isolated trails", "safety_advice"),
    ("Is tap water safe to drink in hotels?", "safety_advice"),
    ("Safety measures when booking houseboats in Kerala", "safety_advice"),
    ("How to protect belongings in crowded train stations?", "safety_advice"),
    ("Precautions for desert safari camping", "safety_advice"),
    ("Safety advice for night walks in tourist towns", "safety_advice"),
    ("How to verify authorized taxis at airports and stations?", "safety_advice"),
    ("What safety measures are needed for river rafting in Rishikesh?", "safety_advice"),
    ("General security advice for international and domestic tourists", "safety_advice"),

    # 24. general_travel
    ("Tell me about tourism in India", "general_travel"),
    ("What makes Kerala special for tourists?", "general_travel"),
    ("Can you explain why Jaipur is called the Pink City?", "general_travel"),
    ("Information about Indian tourism seasons", "general_travel"),
    ("Why is Matheran an automobile-free hill station?", "general_travel"),
    ("Tell me about the culture of Rajasthan", "general_travel"),
    ("What is unique about Hampi ruins?", "general_travel"),
    ("General information on traveling across Maharashtra", "general_travel"),
    ("Why is Goa so popular among international travelers?", "general_travel"),
    ("Explain the backwaters phenomenon of Alleppey", "general_travel"),
    ("What is the significance of the Golden Temple?", "general_travel"),
    ("Tell me interesting facts about Varanasi ghats", "general_travel"),
    ("Overview of travel in Himachal Pradesh", "general_travel"),
    ("Why is Hyderabad known as the City of Pearls?", "general_travel"),
    ("Information on French heritage in Pondicherry", "general_travel"),
    ("What makes Ladakh landscape unique in the world?", "general_travel"),
    ("Tell me about hill stations in the Western Ghats", "general_travel"),
    ("General advice on holidaying in India", "general_travel"),
    ("Tell me about Indian street food culture", "general_travel"),
    ("How many UNESCO heritage sites are in India?", "general_travel"),

    # 25. greeting
    ("Hello", "greeting"),
    ("Hi", "greeting"),
    ("Hey there", "greeting"),
    ("Good morning", "greeting"),
    ("Good evening", "greeting"),
    ("Namaste", "greeting"),
    ("Hi TravelMate", "greeting"),
    ("Hello assistant", "greeting"),
    ("Greetings", "greeting"),
    ("Hey! How are you?", "greeting"),
    ("Hi there!", "greeting"),
    ("Hello there!", "greeting"),
    ("Hey buddy", "greeting"),
    ("Good afternoon", "greeting"),
    ("Hi Travel Assistant", "greeting"),

    # 26. help
    ("What can you do?", "help"),
    ("Help me understand your features", "help"),
    ("How do I use TravelMate?", "help"),
    ("What questions can I ask you?", "help"),
    ("Show me your capabilities", "help"),
    ("Can you guide me on how to use this app?", "help"),
    ("I need help with this travel chatbot", "help"),
    ("What travel assistance do you provide?", "help"),
    ("List everything you can help me with", "help"),
    ("What features does TravelMate offer?", "help"),
    ("Help", "help"),
    ("How does this system work?", "help"),
    ("Can you help me plan my vacation?", "help"),
    ("What kinds of travel queries can I ask?", "help"),
    ("Instructions on using the travel assistant", "help")
]


def train_and_evaluate_model():
    """
    Trains TF-IDF + Logistic Regression intent classifier,
    evaluates on test set, computes real metrics, and saves model artifacts.
    """
    print("=" * 60)
    print("TravelMate Intent Classification Model Training")
    print("=" * 60)

    # 1. Prepare Base DataFrame
    all_examples = []
    for item in INTENT_DATA:
        all_examples.append({
            "query": item[0],
            "intent": item[1],
            "source": "curated_paraphrase"
        })

    # Ingest external public queries from cyberblip/Travel_india
    hf_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "travel_india_hf_raw.csv")
    if os.path.exists(hf_path):
        try:
            df_hf = pd.read_csv(hf_path)
            from collections import defaultdict
            by_intent = defaultdict(list)
            for q in df_hf["input "].dropna():
                ql = str(q).strip().lower()
                if "popular dishes" in ql or "food" in ql:
                    by_intent["food_recommendation"].append(str(q).strip())
                elif "best time" in ql or "when to visit" in ql:
                    by_intent["best_time_to_visit"].append(str(q).strip())
                elif "itinerary" in ql or "plan a" in ql:
                    by_intent["itinerary_planning"].append(str(q).strip())
                elif "places to visit" in ql or "attractions" in ql:
                    by_intent["attraction_recommendation"].append(str(q).strip())
                elif "activities" in ql:
                    by_intent["activity_recommendation"].append(str(q).strip())

            for intent_name, q_list in by_intent.items():
                for q in q_list[:25]:
                    all_examples.append({
                        "query": q,
                        "intent": intent_name,
                        "source": "external_dataset_cyberblip_hf"
                    })
        except Exception as e:
            print(f"Note on external HF data: {e}")

    df = pd.DataFrame(all_examples).drop_duplicates(subset=["query"], keep="first")
    print(f"Total training examples: {len(df)}")
    print(f"Total unique intents: {df['intent'].nunique()}")
    print("Sources breakdown:\n", df["source"].value_counts().to_string())

    # Save training dataset for transparency
    csv_train_path = os.path.join(MODEL_DIR, "intent_training_data.csv")
    df.to_csv(csv_train_path, index=False)
    print(f"Saved dataset snapshot -> {csv_train_path}")

    # 2. Text preprocessing
    df["cleaned_query"] = df["query"].apply(clean_text)

    # 3. Train-Test Split (80% train, 20% test with stratification)
    X = df["cleaned_query"]
    y = df["intent"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training split: {len(X_train)} samples | Test split: {len(X_test)} samples")

    # 4. TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3500,
        sublinear_tf=True,
        min_df=1
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"TF-IDF Vocabulary Size: {len(vectorizer.vocabulary_)} n-grams")

    # 5. Model Training: Logistic Regression with balanced class weights
    classifier = LogisticRegression(
        C=2.5,
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    classifier.fit(X_train_vec, y_train)

    # 6. Evaluation on Unseen Test Split
    y_pred = classifier.predict(X_test_vec)
    accuracy = float(accuracy_score(y_test, y_pred))

    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    labels = sorted(list(set(y)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report_dict = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)

    print("\n--- MODEL EVALUATION RESULTS ---")
    print(f"Accuracy:           {accuracy * 100:.2f}%")
    print(f"Weighted Precision: {precision_w * 100:.2f}%")
    print(f"Weighted Recall:    {recall_w * 100:.2f}%")
    print(f"Weighted F1-Score:  {f1_w * 100:.2f}%")
    print(f"Macro F1-Score:     {f1_m * 100:.2f}%")

    # 7. Save Metrics JSON for Streamlit Model Performance Page
    metrics = {
        "accuracy": round(accuracy, 4),
        "precision_weighted": round(float(precision_w), 4),
        "recall_weighted": round(float(recall_w), 4),
        "f1_weighted": round(float(f1_w), 4),
        "precision_macro": round(float(precision_m), 4),
        "recall_macro": round(float(recall_m), 4),
        "f1_macro": round(float(f1_m), 4),
        "classes": labels,
        "confusion_matrix": cm.tolist(),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "classification_report": report_dict
    }

    metrics_path = os.path.join(MODEL_DIR, "intent_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {metrics_path}")

    # 8. Save Model and Vectorizer Artifacts
    model_path = os.path.join(MODEL_DIR, "intent_classifier.pkl")
    vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

    joblib.dump(classifier, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"Saved model -> {model_path}")
    print(f"Saved vectorizer -> {vec_path}")
    print("=" * 60)
    print("Model Training & Serialization Successfully Finished!")
    print("=" * 60)

    return classifier, vectorizer, metrics


if __name__ == "__main__":
    train_and_evaluate_model()
