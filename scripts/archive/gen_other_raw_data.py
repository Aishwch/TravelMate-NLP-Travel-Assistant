"""
Script to generate attractions_raw.csv, food_raw.csv, budget_raw.csv,
transport_raw.csv, and packing_tips_raw.csv.
"""
import os
import pandas as pd

os.makedirs("data/raw", exist_ok=True)

# 2. Attractions Raw
attractions_data = [
    # Goa
    {"attraction_id": "A001", "attraction_name": "Baga Beach", "destination": "Goa", "city": "North Goa", "state": "Goa",
     "category": "Beach", "description": "Lively beach known for watersports, beach shacks like Britto's, nightlife, and golden sands.",
     "rating": 4.6, "entry_fee": 0, "visit_duration_hours": 3.0, "best_time": "Sunset / Evening", "latitude": 15.5553, "longitude": 73.7517},
    {"attraction_id": "A002", "attraction_name": "Fort Aguada", "destination": "Goa", "city": "Candolim", "state": "Goa",
     "category": "Fort / Historical", "description": "17th-century Portuguese fortress and lighthouse overlooking the vast Arabian Sea.",
     "rating": 4.5, "entry_fee": 50, "visit_duration_hours": 2.0, "best_time": "Morning", "latitude": 15.4920, "longitude": 73.7737},
    {"attraction_id": "A003", "attraction_name": "Basilica of Bom Jesus", "destination": "Goa", "city": "Old Goa", "state": "Goa",
     "category": "Heritage / Church", "description": "UNESCO World Heritage site housing the sacred mortal remains of St. Francis Xavier.",
     "rating": 4.7, "entry_fee": 0, "visit_duration_hours": 1.5, "best_time": "Morning", "latitude": 15.5009, "longitude": 73.9116},
    {"attraction_id": "A004", "attraction_name": "Dudhsagar Falls", "destination": "Goa", "city": "Sonaulim", "state": "Goa",
     "category": "Waterfall / Nature", "description": "Four-tiered majestic 'Sea of Milk' waterfall surrounded by dense tropical deciduous forests.",
     "rating": 4.8, "entry_fee": 110, "visit_duration_hours": 4.0, "best_time": "Morning", "latitude": 15.3144, "longitude": 74.3143},
    {"attraction_id": "A005", "attraction_name": "Palolem Beach", "destination": "Goa", "city": "Canacona", "state": "Goa",
     "category": "Beach / Peaceful", "description": "Crescent-shaped serene beach with calm waters, palm trees, and silent noise parties in South Goa.",
     "rating": 4.8, "entry_fee": 0, "visit_duration_hours": 3.5, "best_time": "Sunset / Evening", "latitude": 15.0100, "longitude": 74.0232},

    # Jaipur
    {"attraction_id": "A006", "attraction_name": "Amber Fort", "destination": "Jaipur", "city": "Jaipur", "state": "Rajasthan",
     "category": "Fort / Historical", "description": "Opulent hilltop fort palace featuring Sheesh Mahal (Mirror Palace) and Maota lake vistas.",
     "rating": 4.8, "entry_fee": 100, "visit_duration_hours": 3.0, "best_time": "Morning", "latitude": 26.9855, "longitude": 75.8513},
    {"attraction_id": "A007", "attraction_name": "Hawa Mahal", "destination": "Jaipur", "city": "Jaipur", "state": "Rajasthan",
     "category": "Heritage / Architecture", "description": "Iconic Palace of Winds with 953 honeycomb windows designed for royal ladies to view street life.",
     "rating": 4.6, "entry_fee": 50, "visit_duration_hours": 1.5, "best_time": "Morning", "latitude": 26.9239, "longitude": 75.8267},
    {"attraction_id": "A008", "attraction_name": "City Palace Jaipur", "destination": "Jaipur", "city": "Jaipur", "state": "Rajasthan",
     "category": "Palace / Museum", "description": "Sprawling royal residence blending Rajput, Mughal, and European architecture with royal museum.",
     "rating": 4.6, "entry_fee": 300, "visit_duration_hours": 2.5, "best_time": "Afternoon", "latitude": 26.9258, "longitude": 75.8237},
    {"attraction_id": "A009", "attraction_name": "Jantar Mantar", "destination": "Jaipur", "city": "Jaipur", "state": "Rajasthan",
     "category": "UNESCO / Science", "description": "World's largest stone astronomical observatory built by Maharaja Sawai Jai Singh II.",
     "rating": 4.5, "entry_fee": 50, "visit_duration_hours": 1.5, "best_time": "Afternoon", "latitude": 26.9248, "longitude": 75.8246},
    {"attraction_id": "A010", "attraction_name": "Nahargarh Fort", "destination": "Jaipur", "city": "Jaipur", "state": "Rajasthan",
     "category": "Fort / Viewpoint", "description": "Edge of the Aravalli Hills overlooking the entire Pink City; famous for magical sunsets.",
     "rating": 4.7, "entry_fee": 50, "visit_duration_hours": 2.0, "best_time": "Sunset / Evening", "latitude": 26.9373, "longitude": 75.8156},

    # Lonavala
    {"attraction_id": "A011", "attraction_name": "Tiger's Leap (Tiger Point)", "destination": "Lonavala", "city": "Lonavala", "state": "Maharashtra",
     "category": "Viewpoint / Nature", "description": "Cliff with a 650m drop that resembles a leaping tiger, offering views of misty valleys and waterfalls.",
     "rating": 4.4, "entry_fee": 0, "visit_duration_hours": 2.0, "best_time": "Morning", "latitude": 18.7360, "longitude": 73.3768},
    {"attraction_id": "A012", "attraction_name": "Bhushi Dam", "destination": "Lonavala", "city": "Lonavala", "state": "Maharashtra",
     "category": "Water / Nature", "description": "Popular masonry dam where monsoon water cascades over stepped rocks, creating natural water splash steps.",
     "rating": 4.1, "entry_fee": 0, "visit_duration_hours": 2.0, "best_time": "Afternoon", "latitude": 18.7303, "longitude": 73.4001},
    {"attraction_id": "A013", "attraction_name": "Karla Caves", "destination": "Lonavala", "city": "Karli", "state": "Maharashtra",
     "category": "Caves / Historical", "description": "Ancient 2nd-century BC rock-cut Buddhist chaitya with intricately carved pillars and stupa.",
     "rating": 4.5, "entry_fee": 25, "visit_duration_hours": 2.0, "best_time": "Morning", "latitude": 18.7825, "longitude": 73.4705},
    {"attraction_id": "A014", "attraction_name": "Pawna Lake", "destination": "Lonavala", "city": "Kamshet", "state": "Maharashtra",
     "category": "Lake / Camping", "description": "Scenic artificial reservoir surrounded by Tung and Tikona forts, premier spot for lakeside camping.",
     "rating": 4.6, "entry_fee": 0, "visit_duration_hours": 4.0, "best_time": "Sunset / Evening", "latitude": 18.6750, "longitude": 73.4860},

    # Matheran
    {"attraction_id": "A015", "attraction_name": "Louisa Point", "destination": "Matheran", "city": "Matheran", "state": "Maharashtra",
     "category": "Viewpoint / Nature", "description": "Panoramic clifftop view overlooking Vishalgad, Prabal fort, and lush valleys with zero vehicular noise.",
     "rating": 4.7, "entry_fee": 0, "visit_duration_hours": 2.0, "best_time": "Sunset / Evening", "latitude": 18.9830, "longitude": 73.2550},
    {"attraction_id": "A016", "attraction_name": "Charlotte Lake", "destination": "Matheran", "city": "Matheran", "state": "Maharashtra",
     "category": "Lake / Nature", "description": "Serene freshwater lake surrounded by dense evergreen woods, echoing bird songs and Pisarnath temple.",
     "rating": 4.5, "entry_fee": 0, "visit_duration_hours": 1.5, "best_time": "Morning", "latitude": 18.9818, "longitude": 73.2678},
    {"attraction_id": "A017", "attraction_name": "Panorama Point", "destination": "Matheran", "city": "Matheran", "state": "Maharashtra",
     "category": "Viewpoint / Sunrise", "description": "Sunrise Point offering a 360-degree panorama of the Western Ghats and Sahyadri mountain chains.",
     "rating": 4.6, "entry_fee": 0, "visit_duration_hours": 2.0, "best_time": "Morning", "latitude": 19.0105, "longitude": 73.2720},
    {"attraction_id": "A018", "attraction_name": "Echo Point", "destination": "Matheran", "city": "Matheran", "state": "Maharashtra",
     "category": "Viewpoint / Nature", "description": "Natural acoustic cliff that reflects sound back across the mist-shrouded gorge.",
     "rating": 4.3, "entry_fee": 0, "visit_duration_hours": 1.0, "best_time": "Afternoon", "latitude": 18.9840, "longitude": 73.2730},

    # Mumbai
    {"attraction_id": "A019", "attraction_name": "Gateway of India", "destination": "Mumbai", "city": "Mumbai", "state": "Maharashtra",
     "category": "Heritage / Monument", "description": "Indo-Saracenic basalt arch built to commemorate the 1911 visit of King George V, overlooking Mumbai Harbour.",
     "rating": 4.7, "entry_fee": 0, "visit_duration_hours": 1.5, "best_time": "Morning", "latitude": 18.9220, "longitude": 72.8347},
    {"attraction_id": "A020", "attraction_name": "Marine Drive", "destination": "Mumbai", "city": "Mumbai", "state": "Maharashtra",
     "category": "Promenade / Coastal", "description": "3.6 km long arc promenade along the Arabian Sea, famously known as the Queen's Necklace at night.",
     "rating": 4.8, "entry_fee": 0, "visit_duration_hours": 2.0, "best_time": "Sunset / Evening", "latitude": 18.9432, "longitude": 72.8230},
    {"attraction_id": "A021", "attraction_name": "Elephanta Caves", "destination": "Mumbai", "city": "Gharapuri", "state": "Maharashtra",
     "category": "UNESCO / Rock-cut Caves", "description": "Island caves featuring the monumental 20-foot three-headed Sadashiva sculpture carved in basalt rock.",
     "rating": 4.6, "entry_fee": 40, "visit_duration_hours": 3.5, "best_time": "Morning", "latitude": 18.9633, "longitude": 72.9315},
    {"attraction_id": "A022", "attraction_name": "Chhatrapati Shivaji Maharaj Terminus", "destination": "Mumbai", "city": "Mumbai", "state": "Maharashtra",
     "category": "UNESCO / Architecture", "description": "Masterpiece of Victorian Gothic Revival architecture blended with traditional Indian styles.",
     "rating": 4.7, "entry_fee": 0, "visit_duration_hours": 1.0, "best_time": "Sunset / Evening", "latitude": 18.9400, "longitude": 72.8354},

    # Mahabaleshwar
    {"attraction_id": "A023", "attraction_name": "Arthur's Seat", "destination": "Mahabaleshwar", "city": "Mahabaleshwar", "state": "Maharashtra",
     "category": "Viewpoint / Nature", "description": "The 'Queen of all Points' offering a dramatic view of the Savitri river valley and Brahma Arayan territory.",
     "rating": 4.6, "entry_fee": 0, "visit_duration_hours": 1.5, "best_time": "Morning", "latitude": 17.9863, "longitude": 73.6214},
    {"attraction_id": "A024", "attraction_name": "Venna Lake", "destination": "Mahabaleshwar", "city": "Mahabaleshwar", "state": "Maharashtra",
     "category": "Lake / Boating", "description": "Picturesque lake flanked by pine trees where visitors enjoy rowboats, paddle boats, and fresh corn.",
     "rating": 4.3, "entry_fee": 0, "visit_duration_hours": 2.0, "best_time": "Sunset / Evening", "latitude": 17.9250, "longitude": 73.6580},
    {"attraction_id": "A025", "attraction_name": "Pratapgad Fort", "destination": "Mahabaleshwar", "city": "Mahabaleshwar", "state": "Maharashtra",
     "category": "Fort / Historical", "description": "Historic hill fortress built by Chhatrapati Shivaji Maharaj, the site of the Battle of Pratapgad.",
     "rating": 4.7, "entry_fee": 0, "visit_duration_hours": 3.0, "best_time": "Morning", "latitude": 17.9257, "longitude": 73.5786},
    {"attraction_id": "A026", "attraction_name": "Mapro Garden", "destination": "Mahabaleshwar", "city": "Panchgani", "state": "Maharashtra",
     "category": "Agri-Tourism / Food", "description": "Strawberry processing park famous for freshly harvested strawberry cream, wood-fired pizza, and fruit crushes.",
     "rating": 4.6, "entry_fee": 0, "visit_duration_hours": 2.0, "best_time": "Afternoon", "latitude": 17.9230, "longitude": 73.7430},

    # Gokarna
    {"attraction_id": "A027", "attraction_name": "Om Beach", "destination": "Gokarna", "city": "Gokarna", "state": "Karnataka",
     "category": "Beach / Nature", "description": "Naturally shaped like the auspicious spiritual symbol 'Om', with two semi-circular coves and rock promontories.",
     "rating": 4.7, "entry_fee": 0, "visit_duration_hours": 3.0, "best_time": "Sunset / Evening", "latitude": 14.5186, "longitude": 74.3168},
    {"attraction_id": "A028", "attraction_name": "Kudle Beach", "destination": "Gokarna", "city": "Gokarna", "state": "Karnataka",
     "category": "Beach / Peaceful", "description": "Wide, white-sand cove ideal for barefoot morning walks, yoga, beach volleyball, and shack dining.",
     "rating": 4.6, "entry_fee": 0, "visit_duration_hours": 3.0, "best_time": "Morning", "latitude": 14.5290, "longitude": 74.3140},
    {"attraction_id": "A029", "attraction_name": "Mahabaleshwar Temple Gokarna", "destination": "Gokarna", "city": "Gokarna", "state": "Karnataka",
     "category": "Spiritual / Temple", "description": "Ancient 4th-century CE temple enshrining the sacred Atmalinga of Lord Shiva, a major Dravidian pilgrimage site.",
     "rating": 4.7, "entry_fee": 0, "visit_duration_hours": 1.5, "best_time": "Morning", "latitude": 14.5428, "longitude": 74.3181},
    {"attraction_id": "A030", "attraction_name": "Paradise Beach", "destination": "Gokarna", "city": "Gokarna", "state": "Karnataka",
     "category": "Beach / Offbeat", "description": "Secluded, raw beach reachable by coastal cliff trek or fisherman boat, devoid of commercial clamor.",
     "rating": 4.8, "entry_fee": 0, "visit_duration_hours": 2.5, "best_time": "Sunset / Evening", "latitude": 14.5020, "longitude": 74.3310},

    # Udaipur
    {"attraction_id": "A031", "attraction_name": "City Palace Udaipur", "destination": "Udaipur", "city": "Udaipur", "state": "Rajasthan",
     "category": "Royal Palace", "description": "Monumental white palace complex on the banks of Lake Pichola with ornate courtyards, balconies, and mirrorwork.",
     "rating": 4.8, "entry_fee": 300, "visit_duration_hours": 3.0, "best_time": "Morning", "latitude": 24.5764, "longitude": 73.6835},
    {"attraction_id": "A032", "attraction_name": "Lake Pichola Boat Cruise", "destination": "Udaipur", "city": "Udaipur", "state": "Rajasthan",
     "category": "Lake / Romantic", "description": "Scenic boat cruise gliding past Taj Lake Palace, Jag Mandir, and royal ghats bathed in golden hour light.",
     "rating": 4.9, "entry_fee": 400, "visit_duration_hours": 1.5, "best_time": "Sunset / Evening", "latitude": 24.5700, "longitude": 73.6780},
    {"attraction_id": "A033", "attraction_name": "Sajjangarh (Monsoon Palace)", "destination": "Udaipur", "city": "Udaipur", "state": "Rajasthan",
     "category": "Palace / Sunset Viewpoint", "description": "Hilltop palace built on Bansdara mountain peak providing sweeping views of the city lakes and Aravali hills.",
     "rating": 4.6, "entry_fee": 110, "visit_duration_hours": 2.0, "best_time": "Sunset / Evening", "latitude": 24.5939, "longitude": 73.6393},
    {"attraction_id": "A034", "attraction_name": "Saheliyon-ki-Bari", "destination": "Udaipur", "city": "Udaipur", "state": "Rajasthan",
     "category": "Garden / Heritage", "description": "Historic Courtyard of Maidens with marble elephants, fountains, lotus pools, and lush green lawns.",
     "rating": 4.5, "entry_fee": 50, "visit_duration_hours": 1.5, "best_time": "Morning", "latitude": 24.6019, "longitude": 73.6872},

    # Munnar
    {"attraction_id": "A035", "attraction_name": "Tea Gardens Munnar", "destination": "Munnar", "city": "Munnar", "state": "Kerala",
     "category": "Nature / Plantation", "description": "Endless rolling hills covered in manicured emerald green tea shrubs with fresh mountain aroma.",
     "rating": 4.9, "entry_fee": 0, "visit_duration_hours": 2.5, "best_time": "Morning", "latitude": 10.0889, "longitude": 77.0595},
    {"attraction_id": "A036", "attraction_name": "Eravikulam National Park", "destination": "Munnar", "city": "Munnar", "state": "Kerala",
     "category": "Wildlife / National Park", "description": "Sanctuary home to the endangered Nilgiri Tahr mountain goat and blooming Neelakurinji flower.",
     "rating": 4.7, "entry_fee": 200, "visit_duration_hours": 3.0, "best_time": "Morning", "latitude": 10.1500, "longitude": 77.0667},
    {"attraction_id": "A037", "attraction_name": "Mattupetty Dam", "destination": "Munnar", "city": "Munnar", "state": "Kerala",
     "category": "Dam / Water Sports", "description": "Concrete gravity dam flanked by tea plantations, offering speed boating and wild elephant sightings.",
     "rating": 4.4, "entry_fee": 50, "visit_duration_hours": 2.0, "best_time": "Afternoon", "latitude": 10.1067, "longitude": 77.1245},
    {"attraction_id": "A038", "attraction_name": "Attukad Waterfalls", "destination": "Munnar", "city": "Munnar", "state": "Kerala",
     "category": "Waterfall / Scenic", "description": "Rushing cascade tucked between thick jungle hills, accessible via an atmospheric wooden bridge.",
     "rating": 4.6, "entry_fee": 0, "visit_duration_hours": 1.5, "best_time": "Sunset / Evening", "latitude": 10.0528, "longitude": 77.0450},

    # Hyderabad
    {"attraction_id": "A039", "attraction_name": "Charminar", "destination": "Hyderabad", "city": "Hyderabad", "state": "Telangana",
     "category": "Monument / Heritage", "description": "1591 monument with four towering minarets, marking the heart of old Hyderabad and bustling Laad Bazaar.",
     "rating": 4.6, "entry_fee": 25, "visit_duration_hours": 1.5, "best_time": "Sunset / Evening", "latitude": 17.3616, "longitude": 78.4747},
    {"attraction_id": "A040", "attraction_name": "Golconda Fort", "destination": "Hyderabad", "city": "Hyderabad", "state": "Telangana",
     "category": "Fort / Acoustic Wonder", "description": "Medieval citadel famous for acoustic engineering where a clap at Fateh Darwaza is heard 1 km away at Bala Hissar.",
     "rating": 4.7, "entry_fee": 25, "visit_duration_hours": 3.0, "best_time": "Afternoon", "latitude": 17.3833, "longitude": 78.4011},
    {"attraction_id": "A041", "attraction_name": "Salar Jung Museum", "destination": "Hyderabad", "city": "Hyderabad", "state": "Telangana",
     "category": "Museum / Art", "description": "One of the world's premier one-man art collections, featuring the Veiled Rebecca statue and Musical Clock.",
     "rating": 4.6, "entry_fee": 50, "visit_duration_hours": 3.0, "best_time": "Morning", "latitude": 17.3713, "longitude": 78.4804},
    {"attraction_id": "A042", "attraction_name": "Chowmahalla Palace", "destination": "Hyderabad", "city": "Hyderabad", "state": "Telangana",
     "category": "Palace / Royal Heritage", "description": "Opulent palace of the Nizams of Hyderabad, featuring vintage Rolls Royce cars and crystal chandeliers.",
     "rating": 4.7, "entry_fee": 100, "visit_duration_hours": 2.0, "best_time": "Morning", "latitude": 17.3578, "longitude": 78.4717},

    # Manali
    {"attraction_id": "A043", "attraction_name": "Solang Valley", "destination": "Manali", "city": "Manali", "state": "Himachal Pradesh",
     "category": "Adventure / Snow", "description": "Year-round adventure hub offering skiing, paragliding, zorbing, snow scooter rides, and cable car ropeways.",
     "rating": 4.7, "entry_fee": 0, "visit_duration_hours": 4.0, "best_time": "Morning", "latitude": 32.3166, "longitude": 77.1578},
    {"attraction_id": "A044", "attraction_name": "Hadimba Temple", "destination": "Manali", "city": "Manali", "state": "Himachal Pradesh",
     "category": "Temple / Wood Architecture", "description": "Unique 1553 pagoda-style wooden temple dedicated to Hadimba Devi, surrounded by towering deodar cedar trees.",
     "rating": 4.6, "entry_fee": 0, "visit_duration_hours": 1.5, "best_time": "Morning", "latitude": 32.2483, "longitude": 77.1804},
    {"attraction_id": "A045", "attraction_name": "Rohtang Pass", "destination": "Manali", "city": "Manali", "state": "Himachal Pradesh",
     "category": "Mountain Pass / Snow", "description": "High mountain pass at 13,058 ft connecting Kullu with Lahaul and Spiti, featuring permanent glaciers and snow points.",
     "rating": 4.8, "entry_fee": 500, "visit_duration_hours": 5.0, "best_time": "Morning", "latitude": 32.3716, "longitude": 77.2466},
    {"attraction_id": "A046", "attraction_name": "Jogini Waterfalls", "destination": "Manali", "city": "Vashisht", "state": "Himachal Pradesh",
     "category": "Waterfall / Trekking", "description": "Scenic short mountain trek from Vashisht village through apple orchards to a cascading stream.",
     "rating": 4.7, "entry_fee": 0, "visit_duration_hours": 2.5, "best_time": "Afternoon", "latitude": 32.2680, "longitude": 77.1950},

    # Hampi
    {"attraction_id": "A047", "attraction_name": "Virupaksha Temple", "destination": "Hampi", "city": "Hampi", "state": "Karnataka",
     "category": "Spiritual / UNESCO", "description": "Seventh-century working Shiva temple with an imposing 50-meter gopuram, holy elephant Lakshmi, and Tungabhadra views.",
     "rating": 4.8, "entry_fee": 25, "visit_duration_hours": 2.0, "best_time": "Morning", "latitude": 15.3350, "longitude": 76.4600},
    {"attraction_id": "A048", "attraction_name": "Vijaya Vittala Temple & Stone Chariot", "destination": "Hampi", "city": "Hampi", "state": "Karnataka",
     "category": "UNESCO / Monument", "description": "The pinnacle of Vijayanagara architecture featuring musical stone pillars and the iconic monolithic Garuda stone chariot.",
     "rating": 4.9, "entry_fee": 40, "visit_duration_hours": 2.5, "best_time": "Afternoon", "latitude": 15.3400, "longitude": 76.4750},
    {"attraction_id": "A049", "attraction_name": "Matanga Hill", "destination": "Hampi", "city": "Hampi", "state": "Karnataka",
     "category": "Viewpoint / Trekking", "description": "The highest point in Hampi providing a breathtaking 360-degree panorama of ancient ruins and boulder landscapes at sunrise.",
     "rating": 4.8, "entry_fee": 0, "visit_duration_hours": 2.0, "best_time": "Morning", "latitude": 15.3320, "longitude": 76.4670},

    # Leh Ladakh
    {"attraction_id": "A050", "attraction_name": "Pangong Tso Lake", "destination": "Leh Ladakh", "city": "Leh", "state": "Ladakh",
     "category": "Alpine Lake / Nature", "description": "High-altitude endorheic lake at 14,270 ft extending into Tibet, renowned for shifting blue hues from azure to turquoise.",
     "rating": 4.9, "entry_fee": 0, "visit_duration_hours": 4.0, "best_time": "Afternoon", "latitude": 33.7595, "longitude": 78.6674},
    {"attraction_id": "A051", "attraction_name": "Nubra Valley & Hunder Sand Dunes", "destination": "Leh Ladakh", "city": "Diskit", "state": "Ladakh",
     "category": "Desert / Valley", "description": "High-altitude cold desert valley famous for white sand dunes and double-humped Bactrian camel rides.",
     "rating": 4.8, "entry_fee": 0, "visit_duration_hours": 4.0, "best_time": "Sunset / Evening", "latitude": 34.5833, "longitude": 77.5667},
    {"attraction_id": "A052", "attraction_name": "Thiksey Monastery", "destination": "Leh Ladakh", "city": "Thiksey", "state": "Ladakh",
     "category": "Monastery / Buddhist", "description": "Twelve-story Gompa resembling Potala Palace of Lhasa, housing a 49-foot high statue of Maitreya Future Buddha.",
     "rating": 4.8, "entry_fee": 30, "visit_duration_hours": 2.0, "best_time": "Morning", "latitude": 34.0583, "longitude": 77.6667},

    # Rishikesh
    {"attraction_id": "A053", "attraction_name": "Triveni Ghat Evening Aarti", "destination": "Rishikesh", "city": "Rishikesh", "state": "Uttarakhand",
     "category": "Spiritual / Ceremony", "description": "Mesmerizing evening prayer ritual with Vedic chants, roaring conch shells, and thousands of floating leaf lamps on the Ganga.",
     "rating": 4.8, "entry_fee": 0, "visit_duration_hours": 1.5, "best_time": "Sunset / Evening", "latitude": 30.1030, "longitude": 78.2980},
    {"attraction_id": "A054", "attraction_name": "Ram Jhula & Laxman Jhula", "destination": "Rishikesh", "city": "Rishikesh", "state": "Uttarakhand",
     "category": "Bridge / Landmark", "description": "Iconic iron suspension footbridges across the holy Ganges connecting temples, ashrams, and riverside cafes.",
     "rating": 4.6, "entry_fee": 0, "visit_duration_hours": 1.5, "best_time": "Morning", "latitude": 30.1250, "longitude": 78.3150},
    {"attraction_id": "A055", "attraction_name": "Beatles Ashram (Chaurasi Kutia)", "destination": "Rishikesh", "city": "Rishikesh", "state": "Uttarakhand",
     "category": "Heritage / Music", "description": "Former ashram of Maharishi Mahesh Yogi where The Beatles composed the White Album, adorned with graffiti art.",
     "rating": 4.5, "entry_fee": 150, "visit_duration_hours": 2.0, "best_time": "Afternoon", "latitude": 30.1160, "longitude": 78.3120},

    # Varanasi
    {"attraction_id": "A056", "attraction_name": "Dashashwamedh Ghat", "destination": "Varanasi", "city": "Varanasi", "state": "Uttar Pradesh",
     "category": "Ghat / Spiritual", "description": "The main and most historic ghat of Varanasi, famous for its grand synchronized evening Ganga Maha Aarti.",
     "rating": 4.9, "entry_fee": 0, "visit_duration_hours": 2.0, "best_time": "Sunset / Evening", "latitude": 25.3076, "longitude": 83.0104},
    {"attraction_id": "A057", "attraction_name": "Kashi Vishwanath Temple", "destination": "Varanasi", "city": "Varanasi", "state": "Uttar Pradesh",
     "category": "Spiritual / Jyotirlinga", "description": "One of the twelve revered Jyotirlinga Shiva temples, crowned with golden domes and renovated corridor.",
     "rating": 4.8, "entry_fee": 0, "visit_duration_hours": 2.0, "best_time": "Morning", "latitude": 25.3109, "longitude": 83.0107},
    {"attraction_id": "A058", "attraction_name": "Sarnath Buddhist Site", "destination": "Varanasi", "city": "Sarnath", "state": "Uttar Pradesh",
     "category": "Buddhist Heritage / UNESCO", "description": "Deer park where Gautama Buddha first taught the Dharma; features Dhamek Stupa and Ashoka Pillar.",
     "rating": 4.7, "entry_fee": 25, "visit_duration_hours": 2.5, "best_time": "Morning", "latitude": 25.3811, "longitude": 83.0214},

    # Agra
    {"attraction_id": "A059", "attraction_name": "Taj Mahal", "destination": "Agra", "city": "Agra", "state": "Uttar Pradesh",
     "category": "World Wonder / UNESCO", "description": "The eternal ivory-white marble mausoleum on the Yamuna river built by Emperor Shah Jahan in memory of Mumtaz Mahal.",
     "rating": 4.9, "entry_fee": 50, "visit_duration_hours": 3.0, "best_time": "Morning", "latitude": 27.1751, "longitude": 78.0421},
    {"attraction_id": "A060", "attraction_name": "Agra Fort", "destination": "Agra", "city": "Agra", "state": "Uttar Pradesh",
     "category": "Mughal Fort / UNESCO", "description": "Massive 16th-century red sandstone fortress that served as the primary residence of the Mughal emperors.",
     "rating": 4.7, "entry_fee": 50, "visit_duration_hours": 2.5, "best_time": "Afternoon", "latitude": 27.1795, "longitude": 78.0211},

    # Coorg
    {"attraction_id": "A061", "attraction_name": "Abbey Falls", "destination": "Coorg", "city": "Madikeri", "state": "Karnataka",
     "category": "Waterfall / Nature", "description": "Roaring cascade nestled within private spice estates and coffee plantations, viewed from a suspension bridge.",
     "rating": 4.5, "entry_fee": 15, "visit_duration_hours": 1.5, "best_time": "Morning", "latitude": 12.4533, "longitude": 75.7197},
    {"attraction_id": "A062", "attraction_name": "Namdroling Tibetan Monastery", "destination": "Coorg", "city": "Bylakuppe", "state": "Karnataka",
     "category": "Spiritual / Tibetan", "description": "The Golden Temple of Bylakuppe, second largest Tibetan settlement in India, with 40-foot gilded statues.",
     "rating": 4.8, "entry_fee": 0, "visit_duration_hours": 2.0, "best_time": "Morning", "latitude": 12.4300, "longitude": 75.9600},

    # Ooty
    {"attraction_id": "A063", "attraction_name": "Nilgiri Mountain Railway (Toy Train)", "destination": "Ooty", "city": "Ooty", "state": "Tamil Nadu",
     "category": "UNESCO / Train Ride", "description": "Historic steam locomotive railway chugging through 16 tunnels, 250 bridges, and steep forested ravines.",
     "rating": 4.8, "entry_fee": 100, "visit_duration_hours": 3.5, "best_time": "Morning", "latitude": 11.4100, "longitude": 76.7000},
    {"attraction_id": "A064", "attraction_name": "Ooty Botanical Gardens", "destination": "Ooty", "city": "Ooty", "state": "Tamil Nadu",
     "category": "Garden / Nature", "description": "Sprawling 55-acre terraced garden established in 1848, home to thousands of exotic plant species and a fossil tree trunk.",
     "rating": 4.4, "entry_fee": 40, "visit_duration_hours": 2.0, "best_time": "Afternoon", "latitude": 11.4172, "longitude": 76.7111},

    # Alleppey
    {"attraction_id": "A065", "attraction_name": "Vembanad Lake Backwaters", "destination": "Alleppey", "city": "Alappuzha", "state": "Kerala",
     "category": "Backwaters / Houseboat", "description": "Longest lake in India, the epicentre of Kerala's serene backwater cruises and annual Nehru Trophy Boat Race.",
     "rating": 4.9, "entry_fee": 0, "visit_duration_hours": 5.0, "best_time": "Sunset / Evening", "latitude": 9.5800, "longitude": 76.3800},
    {"attraction_id": "A066", "attraction_name": "Marari Beach", "destination": "Alleppey", "city": "Mararikulam", "state": "Kerala",
     "category": "Beach / Peaceful", "description": "Clean, secluded coconut-fringed beach offering peaceful golden sands far away from commercial crowds.",
     "rating": 4.7, "entry_fee": 0, "visit_duration_hours": 2.5, "best_time": "Sunset / Evening", "latitude": 9.6000, "longitude": 76.3000},

    # Amritsar
    {"attraction_id": "A067", "attraction_name": "Golden Temple (Sri Harmandir Sahib)", "destination": "Amritsar", "city": "Amritsar", "state": "Punjab",
     "category": "Spiritual / Sikh Heritage", "description": "The holiest Sikh Gurdwara plated in real gold, surrounded by the holy Amrit Sarovar tank and serving free langar to all.",
     "rating": 4.9, "entry_fee": 0, "visit_duration_hours": 3.0, "best_time": "Morning", "latitude": 31.6200, "longitude": 74.8765},
    {"attraction_id": "A068", "attraction_name": "Wagah Border Retreat Ceremony", "destination": "Amritsar", "city": "Attari", "state": "Punjab",
     "category": "Patriotic / Ceremony", "description": "Electrifying daily military parade and flag-lowering ceremony conducted by BSF India and Pakistan Rangers.",
     "rating": 4.8, "entry_fee": 0, "visit_duration_hours": 2.5, "best_time": "Sunset / Evening", "latitude": 31.6042, "longitude": 74.5739},

    # Jaisalmer
    {"attraction_id": "A069", "attraction_name": "Jaisalmer Fort (Sonar Qila)", "destination": "Jaisalmer", "city": "Jaisalmer", "state": "Rajasthan",
     "category": "Fort / Living Heritage", "description": "One of the very few 'living forts' in the world where a quarter of the city's population still resides within golden walls.",
     "rating": 4.8, "entry_fee": 100, "visit_duration_hours": 2.5, "best_time": "Morning", "latitude": 26.9124, "longitude": 70.9126},
    {"attraction_id": "A070", "attraction_name": "Sam Sand Dunes", "destination": "Jaisalmer", "city": "Sam", "state": "Rajasthan",
     "category": "Desert / Dunes", "description": "Classic rolling Thar desert sand dunes where visitors take camel treks, jeep safaris, and camp under the stars.",
     "rating": 4.7, "entry_fee": 0, "visit_duration_hours": 4.0, "best_time": "Sunset / Evening", "latitude": 26.8333, "longitude": 70.5000},

    # Andaman Islands
    {"attraction_id": "A071", "attraction_name": "Radhanagar Beach", "destination": "Andaman Islands", "city": "Havelock Island", "state": "Andaman and Nicobar Islands",
     "category": "Beach / World Class", "description": "Ranked among Asia's best beaches by Time magazine, famous for turquoise waters, white sand, and sunset horizon.",
     "rating": 4.9, "entry_fee": 0, "visit_duration_hours": 3.5, "best_time": "Sunset / Evening", "latitude": 11.9833, "longitude": 92.9500},
    {"attraction_id": "A072", "attraction_name": "Cellular Jail National Memorial", "destination": "Andaman Islands", "city": "Port Blair", "state": "Andaman and Nicobar Islands",
     "category": "History / Memorial", "description": "Historic colonial prison used by the British to exile Indian freedom fighters, known as 'Kala Pani'.",
     "rating": 4.7, "entry_fee": 30, "visit_duration_hours": 2.5, "best_time": "Afternoon", "latitude": 11.6740, "longitude": 92.7483},

    # Pune
    {"attraction_id": "A073", "attraction_name": "Sinhagad Fort", "destination": "Pune", "city": "Pune", "state": "Maharashtra",
     "category": "Fort / Trekking", "description": "Strategic Maratha hill fort famous for Tanaji Malusare's bravery, offering panoramic Sahyadri views and hot pitla bhakri.",
     "rating": 4.6, "entry_fee": 50, "visit_duration_hours": 3.0, "best_time": "Morning", "latitude": 18.3663, "longitude": 73.7558},
    {"attraction_id": "A074", "attraction_name": "Shaniwar Wada", "destination": "Pune", "city": "Pune", "state": "Maharashtra",
     "category": "Palace Fort / Historical", "description": "Historic seat of the Peshwa rulers of the Maratha Empire, famed for its massive fortified Delhi Gate and fountains.",
     "rating": 4.3, "entry_fee": 25, "visit_duration_hours": 1.5, "best_time": "Afternoon", "latitude": 18.5194, "longitude": 73.8553},

    # Alibaug
    {"attraction_id": "A075", "attraction_name": "Kolaba Fort", "destination": "Alibaug", "city": "Alibaug", "state": "Maharashtra",
     "category": "Sea Fort / Heritage", "description": "Ancient 300-year-old Maratha sea fort located 1 km out at sea, reachable on foot at low tide or by horse cart.",
     "rating": 4.4, "entry_fee": 25, "visit_duration_hours": 2.0, "best_time": "Morning", "latitude": 18.6433, "longitude": 72.8569},
    {"attraction_id": "A076", "attraction_name": "Kashid Beach", "destination": "Alibaug", "city": "Kashid", "state": "Maharashtra",
     "category": "Beach / Coastal", "description": "White sand beach framed by Casuarina groves and Arabian sea surf, popular for water scooters and banana rides.",
     "rating": 4.5, "entry_fee": 0, "visit_duration_hours": 3.0, "best_time": "Sunset / Evening", "latitude": 18.4500, "longitude": 72.9000},

    # Shirdi
    {"attraction_id": "A077", "attraction_name": "Sai Baba Samadhi Temple", "destination": "Shirdi", "city": "Shirdi", "state": "Maharashtra",
     "category": "Spiritual / Pilgrimage", "description": "Sacred shrine containing the white Italian marble samadhi of saint Sai Baba, radiating peaceful devotion.",
     "rating": 4.8, "entry_fee": 0, "visit_duration_hours": 2.5, "best_time": "Morning", "latitude": 19.7667, "longitude": 74.4764},
    {"attraction_id": "A078", "attraction_name": "Dwarkamai Mosque", "destination": "Shirdi", "city": "Shirdi", "state": "Maharashtra",
     "category": "Spiritual / Heritage", "description": "Ancient mosque where Sai Baba lived for over 60 years, featuring the perpetual holy dhuni fire.",
     "rating": 4.7, "entry_fee": 0, "visit_duration_hours": 1.0, "best_time": "Afternoon", "latitude": 19.7650, "longitude": 74.4750}
]

df_attr = pd.DataFrame(attractions_data)
df_attr.to_csv("data/raw/attractions_raw.csv", index=False)
print("Saved data/raw/attractions_raw.csv with", len(df_attr), "records")

# 3. Food Raw
food_data = [
    # Goa
    {"food_id": "F001", "destination": "Goa", "state": "Goa", "cuisine": "Goan Coastal & Portuguese",
     "food_item": "Goan Fish Curry & Rice", "description": "Tender kingfish or pomfret simmered in tangy, spicy coconut milk with dried kokum and Kashmiri red chillies.",
     "type": "Non-Veg", "famous_spots": "Vinayak Family Restaurant (Assagao), Ritz Classic (Panaji)", "price_range": "Budget (₹150-₹300)"},
    {"food_id": "F002", "destination": "Goa", "state": "Goa", "cuisine": "Goan Dessert",
     "food_item": "Bebinca", "description": "Traditional 7 to 16 layered Indo-Portuguese pudding baked with coconut milk, sugar, ghee, and egg yolk.",
     "type": "Veg", "famous_spots": "Infantaria Bakery (Baga), Martin's Corner (Betalbatim)", "price_range": "Budget (₹100-₹200)"},
    {"food_id": "F003", "destination": "Goa", "state": "Goa", "cuisine": "Goan Street Snack",
     "food_item": "Pao with Mushroom Xacuti", "description": "Fragrant curry roasted with white poppy seeds, dried coconut, and whole spices served with freshly baked poee bread.",
     "type": "Veg", "famous_spots": "Panaji Church Square Tea Kiosks, Cafe Tato", "price_range": "Budget (₹80-₹150)"},

    # Jaipur
    {"food_id": "F004", "destination": "Jaipur", "state": "Rajasthan", "cuisine": "Rajasthani Traditional",
     "food_item": "Dal Baati Churma", "description": "Crisp baked whole wheat baatis crushed and dipped in pure desi ghee, served with five-lentil spicy dal and sweet crushed churma.",
     "type": "Veg", "famous_spots": "Laxmi Misthan Bhandar (LMB) Johari Bazaar, Chokhi Dhani", "price_range": "Mid-range (₹250-₹500)"},
    {"food_id": "F005", "destination": "Jaipur", "state": "Rajasthan", "cuisine": "Rajasthani Sweet",
     "food_item": "Ghevar & Pyaz Kachori", "description": "Honeycomb disc-shaped sweet soaked in saffron sugar syrup, paired with piping hot crisp onion-filled kachoris.",
     "type": "Veg", "famous_spots": "Rawat Mishthan Bhandar, LMB Jaipur", "price_range": "Budget (₹50-₹120)"},
    {"food_id": "F006", "destination": "Jaipur", "state": "Rajasthan", "cuisine": "Rajasthani Royal Non-Veg",
     "food_item": "Laal Maas", "description": "Fiery royal mutton curry cooked with Mathania red chilies, garlic, and yogurt; historically prepared after royal tiger hunts.",
     "type": "Non-Veg", "famous_spots": "Handi Restaurant (MI Road), Niros", "price_range": "Mid-range (₹400-₹700)"},

    # Mumbai
    {"food_id": "F007", "destination": "Mumbai", "state": "Maharashtra", "cuisine": "Maharashtrian Street Food",
     "food_item": "Vada Pav & Pav Bhaji", "description": "Spicy potato fritter stuffed in fresh pav bun with garlic chutney, and buttery mashed vegetable bhaji with toasted pav.",
     "type": "Veg", "famous_spots": "Ashok Vada Pav (Kirti College), Sardar Pav Bhaji (Tardeo), Cannon Pav Bhaji", "price_range": "Budget (₹30-₹150)"},
    {"food_id": "F008", "destination": "Mumbai", "state": "Maharashtra", "cuisine": "Konkani Coastal",
     "food_item": "Bombil Fry & Surmai Thali", "description": "Crispy semolina (rava) crusted Bombay Duck fish and kingfish thali with coconut solkadhi.",
     "type": "Non-Veg", "famous_spots": "Gajalee (Vile Parle), Mahesh Lunch Home, Highway Gomantak", "price_range": "Mid-range (₹300-₹600)"},
    {"food_id": "F009", "destination": "Mumbai", "state": "Maharashtra", "cuisine": "Parsi Cafe Cuisine",
     "food_item": "Bun Maska & Irani Chai", "description": "Crusty bun slathered with rich salted Amul butter dipped into sweet cardamom-infused Irani milk tea.",
     "type": "Veg", "famous_spots": "Kyani & Co (Marine Lines), Britannia & Co, Cafe Mondegar", "price_range": "Budget (₹50-₹100)"},

    # Hyderabad
    {"food_id": "F010", "destination": "Hyderabad", "state": "Telangana", "cuisine": "Hyderabadi Nizami",
     "food_item": "Hyderabadi Dum Biryani", "description": "Fragrant long-grain basmati rice and marinated goat meat layered with saffron, mint, and spices, cooked on slow dum in a sealed handi.",
     "type": "Non-Veg", "famous_spots": "Paradise (Secunderabad), Bawarchi (RTC X Roads), Cafe Bahar, Shadab", "price_range": "Mid-range (₹250-₹450)"},
    {"food_id": "F011", "destination": "Hyderabad", "state": "Telangana", "cuisine": "Nizami Dessert",
     "food_item": "Double Ka Meetha & Irani Chai", "description": "Royal bread pudding soaked in saffron-infused milk and cardamom, garnished with roasted cashews and silver foil.",
     "type": "Veg", "famous_spots": "Nimrah Cafe near Charminar, Pista House", "price_range": "Budget (₹40-₹100)"},
    {"food_id": "F012", "destination": "Hyderabad", "state": "Telangana", "cuisine": "Hyderabadi Street Special",
     "food_item": "Hyderabadi Haleem", "description": "Slow-cooked savory porridge of pounded meat, lentils, broken wheat, and pure ghee, garnished with fried onions and lemon.",
     "type": "Non-Veg", "famous_spots": "Pista House, Sarvi, Shah Ghouse", "price_range": "Budget (₹150-₹250)"},

    # Lonavala & Matheran
    {"food_id": "F013", "destination": "Lonavala", "state": "Maharashtra", "cuisine": "Maharashtra Sweet & Snack",
     "food_item": "Lonavala Chikki & Walnut Fudge", "description": "Crunchy brittle confection made from jaggery and roasted peanuts, along with velvety chocolate walnut fudge.",
     "type": "Veg", "famous_spots": "Maganlal Chikki, Cooper's Fudge", "price_range": "Budget (₹100-₹300)"},
    {"food_id": "F014", "destination": "Matheran", "state": "Maharashtra", "cuisine": "Maharashtrian Rustic",
     "food_item": "Pithla Bhakri & Thecha", "description": "Wholesome chickpea flour curry served with rustic bajra or jowar flatbread, spicy crushed green chili thecha, and raw onions.",
     "type": "Veg", "famous_spots": "Local dhabas near Charlotte Lake and Matheran Bazaar", "price_range": "Budget (₹80-₹150)"},

    # Mahabaleshwar
    {"food_id": "F015", "destination": "Mahabaleshwar", "state": "Maharashtra", "cuisine": "Fresh Agri-Dessert",
     "food_item": "Fresh Strawberries with Whipped Cream", "description": "Plump handpicked red strawberries smothered in thick fresh sweetened cream and strawberry ice cream.",
     "type": "Veg", "famous_spots": "Mapro Garden (Panchgani road), Bagicha Corner", "price_range": "Budget (₹150-₹250)"},

    # Gokarna
    {"food_id": "F016", "destination": "Gokarna", "state": "Karnataka", "cuisine": "Coastal Kannada & Shack Fusion",
     "food_item": "Prawn Ghee Roast & Neer Dosa", "description": "Succulent prawns roasted in spicy Byadgi chilli paste and pure ghee, paired with paper-thin lace rice crepes.",
     "type": "Non-Veg", "famous_spots": "Namaste Cafe (Om Beach), Chez Christophe (Kudle Beach)", "price_range": "Mid-range (₹200-₹400)"},

    # Munnar & Kerala
    {"food_id": "F017", "destination": "Munnar", "state": "Kerala", "cuisine": "Kerala Traditional",
     "food_item": "Appam with Vegetable / Chicken Stew", "description": "Lacy fermented rice batter hoppers with a fluffy soft center, served with coconut milk aromatic vegetable stew.",
     "type": "Both", "famous_spots": "Rapsy Restaurant, Saravana Bhavan Munnar", "price_range": "Budget (₹100-₹250)"},
    {"food_id": "F018", "destination": "Alleppey", "state": "Kerala", "cuisine": "Kerala Backwater Seafood",
     "food_item": "Karimeen Pollichathu", "description": "Fresh pearl spot fish marinated in shallots, ginger, and spices, wrapped in tender banana leaves and pan-grilled.",
     "type": "Non-Veg", "famous_spots": "Thaff Restaurant, Houseboat onboard dining", "price_range": "Mid-range (₹300-₹550)"},

    # Pune
    {"food_id": "F019", "destination": "Pune", "state": "Maharashtra", "cuisine": "Maharashtrian Puneri",
     "food_item": "Puneri Misal Pav", "description": "Spicy sprouted moth bean curry topped with crunchy farsan mixture, diced onions, lemon, and served with hot buttery pav.",
     "type": "Veg", "famous_spots": "Katakirr Misal (Erandwane), Bedekar Misal, Vaidya Upahar Gruha", "price_range": "Budget (₹60-₹120)"},
    {"food_id": "F020", "destination": "Pune", "state": "Maharashtra", "cuisine": "Puneri Sweet & Drink",
     "food_item": "Mango Mastani & Bakarwadi", "description": "Rich thick mango milkshake topped with ice cream, cherries, and dry fruits, paired with savory Chitale Bandhu spiral snacks.",
     "type": "Veg", "famous_spots": "Sujata Mastani, Chitale Bandhu Mithaiwale", "price_range": "Budget (₹70-₹150)"},

    # Amritsar
    {"food_id": "F021", "destination": "Amritsar", "state": "Punjab", "cuisine": "Punjabi Authentic",
     "food_item": "Amritsari Kulcha with Chole & Lassi", "description": "Crispy, layered tandoor-baked flatbread stuffed with spiced potatoes and onions, served with spicy chickpeas and churned sweet lassi.",
     "type": "Veg", "famous_spots": "Bhai Kulwant Singh Kulchian Wale, Kesar Da Dhaba", "price_range": "Budget (₹80-₹160)"},

    # Varanasi
    {"food_id": "F022", "destination": "Varanasi", "state": "Uttar Pradesh", "cuisine": "Banarasi Street Delicacies",
     "food_item": "Banarasi Kachori Sabzi & Malaiyo", "description": "Crispy hing-flavored kachoris with tangy potato curry, followed by frothy saffron-infused winter milk foam dessert.",
     "type": "Veg", "famous_spots": "Ram Bhandar (Thatheri Bazaar), Blue Lassi Shop, Keshav Tambul", "price_range": "Budget (₹40-₹100)"}
]

df_food = pd.DataFrame(food_data)
df_food.to_csv("data/raw/food_raw.csv", index=False)
print("Saved data/raw/food_raw.csv with", len(df_food), "records")

# 4. Budget Raw
budget_data = [
    {"destination": "Goa", "accommodation_budget": 800, "accommodation_midrange": 2200, "accommodation_luxury": 6500,
     "food_cost_per_day": 700, "local_transport_cost_per_day": 500, "activity_cost_per_day": 800, "daily_budget_min": 1800, "daily_budget_avg": 2800, "budget_tier": "Moderate"},
    {"destination": "Jaipur", "accommodation_budget": 700, "accommodation_midrange": 1800, "accommodation_luxury": 5000,
     "food_cost_per_day": 500, "local_transport_cost_per_day": 400, "activity_cost_per_day": 600, "daily_budget_min": 1500, "daily_budget_avg": 2400, "budget_tier": "Moderate"},
    {"destination": "Lonavala", "accommodation_budget": 600, "accommodation_midrange": 1600, "accommodation_luxury": 4500,
     "food_cost_per_day": 500, "local_transport_cost_per_day": 400, "activity_cost_per_day": 300, "daily_budget_min": 1300, "daily_budget_avg": 2000, "budget_tier": "Budget-Friendly"},
    {"destination": "Matheran", "accommodation_budget": 600, "accommodation_midrange": 1500, "accommodation_luxury": 4000,
     "food_cost_per_day": 450, "local_transport_cost_per_day": 300, "activity_cost_per_day": 250, "daily_budget_min": 1200, "daily_budget_avg": 1800, "budget_tier": "Budget-Friendly"},
    {"destination": "Mumbai", "accommodation_budget": 1000, "accommodation_midrange": 2800, "accommodation_luxury": 8000,
     "food_cost_per_day": 700, "local_transport_cost_per_day": 500, "activity_cost_per_day": 500, "daily_budget_min": 2000, "daily_budget_avg": 3200, "budget_tier": "Premium"},
    {"destination": "Mahabaleshwar", "accommodation_budget": 700, "accommodation_midrange": 1800, "accommodation_luxury": 4800,
     "food_cost_per_day": 550, "local_transport_cost_per_day": 450, "activity_cost_per_day": 350, "daily_budget_min": 1450, "daily_budget_avg": 2200, "budget_tier": "Moderate"},
    {"destination": "Gokarna", "accommodation_budget": 500, "accommodation_midrange": 1300, "accommodation_luxury": 3800,
     "food_cost_per_day": 400, "local_transport_cost_per_day": 300, "activity_cost_per_day": 200, "daily_budget_min": 1100, "daily_budget_avg": 1600, "budget_tier": "Budget-Friendly"},
    {"destination": "Udaipur", "accommodation_budget": 800, "accommodation_midrange": 2200, "accommodation_luxury": 7000,
     "food_cost_per_day": 600, "local_transport_cost_per_day": 400, "activity_cost_per_day": 600, "daily_budget_min": 1600, "daily_budget_avg": 2600, "budget_tier": "Moderate"},
    {"destination": "Munnar", "accommodation_budget": 700, "accommodation_midrange": 1700, "accommodation_luxury": 5000,
     "food_cost_per_day": 500, "local_transport_cost_per_day": 450, "activity_cost_per_day": 350, "daily_budget_min": 1400, "daily_budget_avg": 2100, "budget_tier": "Moderate"},
    {"destination": "Hyderabad", "accommodation_budget": 700, "accommodation_midrange": 1700, "accommodation_luxury": 4500,
     "food_cost_per_day": 550, "local_transport_cost_per_day": 400, "activity_cost_per_day": 350, "daily_budget_min": 1400, "daily_budget_avg": 2200, "budget_tier": "Budget-Friendly"},
    {"destination": "Manali", "accommodation_budget": 700, "accommodation_midrange": 1900, "accommodation_luxury": 5500,
     "food_cost_per_day": 600, "local_transport_cost_per_day": 500, "activity_cost_per_day": 700, "daily_budget_min": 1600, "daily_budget_avg": 2500, "budget_tier": "Moderate"},
    {"destination": "Hampi", "accommodation_budget": 450, "accommodation_midrange": 1200, "accommodation_luxury": 3200,
     "food_cost_per_day": 350, "local_transport_cost_per_day": 250, "activity_cost_per_day": 200, "daily_budget_min": 950, "daily_budget_avg": 1500, "budget_tier": "Budget-Friendly"},
    {"destination": "Pondicherry", "accommodation_budget": 650, "accommodation_midrange": 1600, "accommodation_luxury": 4500,
     "food_cost_per_day": 550, "local_transport_cost_per_day": 350, "activity_cost_per_day": 250, "daily_budget_min": 1350, "daily_budget_avg": 2000, "budget_tier": "Budget-Friendly"},
    {"destination": "Leh Ladakh", "accommodation_budget": 1000, "accommodation_midrange": 2600, "accommodation_luxury": 7500,
     "food_cost_per_day": 750, "local_transport_cost_per_day": 1000, "activity_cost_per_day": 600, "daily_budget_min": 2400, "daily_budget_avg": 3500, "budget_tier": "Premium"},
    {"destination": "Rishikesh", "accommodation_budget": 550, "accommodation_midrange": 1400, "accommodation_luxury": 4200,
     "food_cost_per_day": 450, "local_transport_cost_per_day": 300, "activity_cost_per_day": 500, "daily_budget_min": 1200, "daily_budget_avg": 1800, "budget_tier": "Budget-Friendly"},
    {"destination": "Varanasi", "accommodation_budget": 500, "accommodation_midrange": 1300, "accommodation_luxury": 4000,
     "food_cost_per_day": 400, "local_transport_cost_per_day": 300, "activity_cost_per_day": 300, "daily_budget_min": 1100, "daily_budget_avg": 1700, "budget_tier": "Budget-Friendly"},
    {"destination": "Agra", "accommodation_budget": 700, "accommodation_midrange": 1800, "accommodation_luxury": 5000,
     "food_cost_per_day": 500, "local_transport_cost_per_day": 400, "activity_cost_per_day": 500, "daily_budget_min": 1450, "daily_budget_avg": 2300, "budget_tier": "Moderate"},
    {"destination": "Coorg", "accommodation_budget": 750, "accommodation_midrange": 1800, "accommodation_luxury": 5200,
     "food_cost_per_day": 550, "local_transport_cost_per_day": 450, "activity_cost_per_day": 350, "daily_budget_min": 1500, "daily_budget_avg": 2200, "budget_tier": "Moderate"},
    {"destination": "Ooty", "accommodation_budget": 700, "accommodation_midrange": 1700, "accommodation_luxury": 4800,
     "food_cost_per_day": 500, "local_transport_cost_per_day": 450, "activity_cost_per_day": 300, "daily_budget_min": 1400, "daily_budget_avg": 2100, "budget_tier": "Moderate"},
    {"destination": "Darjeeling", "accommodation_budget": 750, "accommodation_midrange": 1900, "accommodation_luxury": 5200,
     "food_cost_per_day": 550, "local_transport_cost_per_day": 450, "activity_cost_per_day": 400, "daily_budget_min": 1500, "daily_budget_avg": 2300, "budget_tier": "Moderate"},
    {"destination": "Shillong", "accommodation_budget": 700, "accommodation_midrange": 1800, "accommodation_luxury": 5000,
     "food_cost_per_day": 550, "local_transport_cost_per_day": 500, "activity_cost_per_day": 350, "daily_budget_min": 1500, "daily_budget_avg": 2200, "budget_tier": "Moderate"},
    {"destination": "Alleppey", "accommodation_budget": 800, "accommodation_midrange": 2200, "accommodation_luxury": 7500,
     "food_cost_per_day": 650, "local_transport_cost_per_day": 450, "activity_cost_per_day": 800, "daily_budget_min": 1800, "daily_budget_avg": 3000, "budget_tier": "Moderate"},
    {"destination": "Amritsar", "accommodation_budget": 550, "accommodation_midrange": 1400, "accommodation_luxury": 4000,
     "food_cost_per_day": 450, "local_transport_cost_per_day": 300, "activity_cost_per_day": 200, "daily_budget_min": 1150, "daily_budget_avg": 1800, "budget_tier": "Budget-Friendly"},
    {"destination": "Jaisalmer", "accommodation_budget": 700, "accommodation_midrange": 1800, "accommodation_luxury": 5500,
     "food_cost_per_day": 550, "local_transport_cost_per_day": 450, "activity_cost_per_day": 650, "daily_budget_min": 1550, "daily_budget_avg": 2400, "budget_tier": "Moderate"},
    {"destination": "Andaman Islands", "accommodation_budget": 1100, "accommodation_midrange": 2800, "accommodation_luxury": 8500,
     "food_cost_per_day": 800, "local_transport_cost_per_day": 700, "activity_cost_per_day": 1000, "daily_budget_min": 2500, "daily_budget_avg": 3800, "budget_tier": "Premium"},
    {"destination": "Pune", "accommodation_budget": 650, "accommodation_midrange": 1600, "accommodation_luxury": 4500,
     "food_cost_per_day": 450, "local_transport_cost_per_day": 400, "activity_cost_per_day": 250, "daily_budget_min": 1300, "daily_budget_avg": 1900, "budget_tier": "Budget-Friendly"},
    {"destination": "Alibaug", "accommodation_budget": 700, "accommodation_midrange": 1700, "accommodation_luxury": 4800,
     "food_cost_per_day": 550, "local_transport_cost_per_day": 350, "activity_cost_per_day": 300, "daily_budget_min": 1400, "daily_budget_avg": 2000, "budget_tier": "Budget-Friendly"},
    {"destination": "Wayanad", "accommodation_budget": 650, "accommodation_midrange": 1600, "accommodation_luxury": 4600,
     "food_cost_per_day": 500, "local_transport_cost_per_day": 450, "activity_cost_per_day": 350, "daily_budget_min": 1400, "daily_budget_avg": 2000, "budget_tier": "Budget-Friendly"},
    {"destination": "Kochi", "accommodation_budget": 700, "accommodation_midrange": 1700, "accommodation_luxury": 5000,
     "food_cost_per_day": 550, "local_transport_cost_per_day": 400, "activity_cost_per_day": 350, "daily_budget_min": 1450, "daily_budget_avg": 2200, "budget_tier": "Moderate"},
    {"destination": "Dharamshala", "accommodation_budget": 550, "accommodation_midrange": 1400, "accommodation_luxury": 4000,
     "food_cost_per_day": 450, "local_transport_cost_per_day": 350, "activity_cost_per_day": 300, "daily_budget_min": 1200, "daily_budget_avg": 1800, "budget_tier": "Budget-Friendly"},
    {"destination": "Nainital", "accommodation_budget": 650, "accommodation_midrange": 1600, "accommodation_luxury": 4500,
     "food_cost_per_day": 500, "local_transport_cost_per_day": 400, "activity_cost_per_day": 350, "daily_budget_min": 1350, "daily_budget_avg": 2000, "budget_tier": "Budget-Friendly"},
    {"destination": "Gangtok", "accommodation_budget": 750, "accommodation_midrange": 1900, "accommodation_luxury": 5500,
     "food_cost_per_day": 600, "local_transport_cost_per_day": 600, "activity_cost_per_day": 450, "daily_budget_min": 1650, "daily_budget_avg": 2500, "budget_tier": "Moderate"},
    {"destination": "Shimla", "accommodation_budget": 750, "accommodation_midrange": 1800, "accommodation_luxury": 5000,
     "food_cost_per_day": 550, "local_transport_cost_per_day": 450, "activity_cost_per_day": 350, "daily_budget_min": 1500, "daily_budget_avg": 2300, "budget_tier": "Moderate"},
    {"destination": "Kanyakumari", "accommodation_budget": 550, "accommodation_midrange": 1400, "accommodation_luxury": 3800,
     "food_cost_per_day": 400, "local_transport_cost_per_day": 300, "activity_cost_per_day": 250, "daily_budget_min": 1150, "daily_budget_avg": 1700, "budget_tier": "Budget-Friendly"},
    {"destination": "Mysore", "accommodation_budget": 600, "accommodation_midrange": 1500, "accommodation_luxury": 4200,
     "food_cost_per_day": 450, "local_transport_cost_per_day": 350, "activity_cost_per_day": 300, "daily_budget_min": 1250, "daily_budget_avg": 1900, "budget_tier": "Budget-Friendly"},
    {"destination": "Shirdi", "accommodation_budget": 450, "accommodation_midrange": 1200, "accommodation_luxury": 3200,
     "food_cost_per_day": 300, "local_transport_cost_per_day": 250, "activity_cost_per_day": 150, "daily_budget_min": 900, "daily_budget_avg": 1400, "budget_tier": "Budget-Friendly"}
]

df_bud = pd.DataFrame(budget_data)
df_bud.to_csv("data/raw/budget_raw.csv", index=False)
print("Saved data/raw/budget_raw.csv with", len(df_bud), "records")

# 5. Transport Raw
transport_data = [
    # From Mumbai
    {"route_id": "T001", "origin": "Mumbai", "destination": "Goa", "distance_km": 590,
     "modes": "Flight, Train, Bus, Self-Drive / Cab",
     "travel_time": "Flight: 1h 15m | Train: 8-11h (Tejas/Vande Bharat/Konkan Kanya) | Sleeper Bus: 12-14h | Drive: 11-13h",
     "estimated_cost_range": "Flight: ₹2200-₹5500 | Train: ₹400-₹1800 | Bus: ₹800-₹2000 | Cab: ₹9000-₹13000",
     "general_route_info": "Konkan Railway is breathtakingly scenic during daylight. Tejas Express and Vande Bharat provide rapid daytime comfort. Overnight sleeper AC buses depart from Borivali, Dadar, and Vashi."},
    {"route_id": "T002", "origin": "Mumbai", "destination": "Lonavala", "distance_km": 83,
     "modes": "Local/Express Train, Cab, Self-Drive, State Bus",
     "travel_time": "Express Train: 1h 45m | Cab / Drive via Mumbai-Pune Expressway: 2h | Bus: 2h 30m",
     "estimated_cost_range": "Train: ₹60-₹150 | Bus: ₹120-₹250 | Shared Cab: ₹350-₹500 | Private Cab: ₹1800-₹2600",
     "general_route_info": "Smooth, scenic drive via the 6-lane Mumbai-Pune Expressway through Bhor Ghat. Trains depart frequently from CSMT, Dadar, and Thane."},
    {"route_id": "T003", "origin": "Mumbai", "destination": "Matheran", "distance_km": 80,
     "modes": "Local Suburban Train + Toy Train / Shared Taxi, Cab to Dasturi Naka",
     "travel_time": "Local Train to Neral: 1h 40m | Neral to Aman Lodge/Dasturi: 25 mins | Walk/Horse to town: 30 mins",
     "estimated_cost_range": "Train: ₹30-₹100 | Shared Cab Neral to Dasturi: ₹100 | Toy Train Shuttle: ₹50 | Horse: ₹400-₹800",
     "general_route_info": "Matheran is Asia's only automobile-free hill station. All vehicles must park at Dasturi Car Park. From Dasturi, you can take the shuttle toy train, ride a horse, or enjoy a 30-min peaceful forest walk into town."},
    {"route_id": "T004", "origin": "Mumbai", "destination": "Mahabaleshwar", "distance_km": 260,
     "modes": "Sleeper Bus, Cab, Self-Drive, Train to Satara",
     "travel_time": "Direct Volvo AC Bus: 5-6h | Car via Pune Expressway & NH48: 5h | Train to Satara + Cab: 6h",
     "estimated_cost_range": "Bus: ₹600-₹1200 | Private Cab: ₹4500-₹6500 | Train + Cab: ₹500-₹1200",
     "general_route_info": "Drive via Mumbai-Pune Expressway to Shirwal and then ascend through scenic Pasarni Ghat via Wai."},
    {"route_id": "T005", "origin": "Mumbai", "destination": "Alibaug", "distance_km": 95,
     "modes": "Speedboat / Ferry from Gateway of India + Bus, Ro-Ro Car Ferry from Bhaucha Dhakka, Road Drive",
     "travel_time": "Ferry from Gateway to Mandwa: 50m (+ 30m bus to Alibaug) | M2M Ro-Ro Ferry: 60m | Road drive: 3h",
     "estimated_cost_range": "Ferry + connecting bus: ₹150-₹250 | Speedboat: ₹600-₹1000 | Ro-Ro Car Ferry: ₹1000-₹1800 | Cab: ₹2500-₹3500",
     "general_route_info": "Taking the catamaran/ferry from Gateway of India to Mandwa jetty is by far the fastest and most scenic route."},
    {"route_id": "T006", "origin": "Mumbai", "destination": "Pune", "distance_km": 150,
     "modes": "Express Train, Vande Bharat, Shivneri AC Bus, Cab / Carpool",
     "travel_time": "Vande Bharat / Deccan Queen: 2h 45m | Expressway Cab/Bus: 3h - 3h 30m",
     "estimated_cost_range": "Train: ₹100-₹650 | Shivneri MSRTC AC Bus: ₹450 | Cab / BlaBlaCar: ₹350-₹2200",
     "general_route_info": "Decades of commuters rely on Deccan Queen or Pragati Express. Road trip on Mumbai-Pune expressway features popular food mall halts."},
    {"route_id": "T007", "origin": "Mumbai", "destination": "Jaipur", "distance_km": 1150,
     "modes": "Flight, Superfast Express Train, Overnight Bus",
     "travel_time": "Flight: 1h 45m | Train (Garib Rath/August Kranti): 14-16h",
     "estimated_cost_range": "Flight: ₹3000-₹6500 | Train: ₹550-₹2200",
     "general_route_info": "Multiple daily direct flights connect Mumbai (BOM) to Jaipur (JAI). Overnight express trains run from Mumbai Central and Bandra Terminus."},
    {"route_id": "T008", "origin": "Mumbai", "destination": "Udaipur", "distance_km": 750,
     "modes": "Flight, Express Train, Sleeper Bus, Road Drive",
     "travel_time": "Flight: 1h 20m | Express Train: 13h | Sleeper Bus: 14h | Drive via NH48: 12-13h",
     "estimated_cost_range": "Flight: ₹2800-₹6000 | Train: ₹450-₹1800 | Bus: ₹900-₹2000",
     "general_route_info": "Direct flights connect BOM and UDR daily. Bandra Terminus - Udaipur Express offers a comfortable overnight journey."},
    {"route_id": "T009", "origin": "Mumbai", "destination": "Gokarna", "distance_km": 700,
     "modes": "Konkan Railway Train, Sleeper Bus, Flight to Goa (GOX/GOI) + Cab",
     "travel_time": "Train to Gokarna Road: 10-12h (Matsyagandha Express) | Sleeper Bus: 13-14h | Flight to Goa + 3h drive: 5h total",
     "estimated_cost_range": "Train: ₹400-₹1600 | Bus: ₹1000-₹2200 | Flight + Cab: ₹3500-₹7500",
     "general_route_info": "Matsyagandha Express runs directly from Lokmanya Tilak Terminus (LTT) to Gokarna Road station."},

    # From Delhi
    {"route_id": "T010", "origin": "Delhi", "destination": "Jaipur", "distance_km": 280,
     "modes": "Vande Bharat, Shatabdi Express, Volvo AC Bus, Delhi-Mumbai Expressway Drive",
     "travel_time": "Vande Bharat / Shatabdi: 3h 15m - 3h 45m | Car via Expressway: 3h 30m | Volvo Bus: 5h",
     "estimated_cost_range": "Train: ₹500-₹1300 | Bus: ₹400-₹900 | Cab: ₹3500-₹5000",
     "general_route_info": "The new Delhi-Mumbai Expressway section has reduced driving time between Sohna and Dausa/Jaipur to around 3.5 hours."},
    {"route_id": "T011", "origin": "Delhi", "destination": "Agra", "distance_km": 210,
     "modes": "Gatimaan Express, Vande Bharat, Yamuna Expressway Drive, Bus",
     "travel_time": "Gatimaan Express: 1h 40m | Yamuna Expressway Drive: 2h 30m - 3h | Bus: 3h 30m",
     "estimated_cost_range": "Train: ₹350-₹1000 | Cab: ₹2500-₹4000 | Bus: ₹300-₹600",
     "general_route_info": "Gatimaan Express from Hazrat Nizamuddin to Agra Cantt is India's premier tourist train. Yamuna Expressway is a high-speed toll road."},
    {"route_id": "T012", "origin": "Delhi", "destination": "Manali", "distance_km": 540,
     "modes": "Overnight Volvo Bus, Self-Drive / Cab, Flight to Bhuntar (KUU)",
     "travel_time": "Volvo Bus: 12-14h | Drive via Kiratpur-Manali highway: 10-12h | Flight to Kullu: 1h 15m",
     "estimated_cost_range": "Bus: ₹1000-₹2200 | Cab: ₹8000-₹13000 | Flight: ₹5000-₹11000",
     "general_route_info": "HPTDC and private AC Volvo buses depart every evening from Kashmere Gate ISBT and Majnu Ka Tila. New 4-lane highway through Kiratpur has significantly improved travel times."},
    {"route_id": "T013", "origin": "Delhi", "destination": "Rishikesh", "distance_km": 240,
     "modes": "Vande Bharat / Jan Shatabdi to Haridwar + Cab, Direct Bus, Drive",
     "travel_time": "Train to Haridwar (3h 30m) + 40m Cab | Drive via Meerut Expressway: 4h 30m | Volvo Bus: 5-6h",
     "estimated_cost_range": "Train: ₹450-₹1200 | Bus: ₹400-₹900 | Cab: ₹3500-₹5000",
     "general_route_info": "Delhi-Meerut Expressway has cut travel time considerably. Haridwar is the major rail junction connecting Rishikesh."},
    {"route_id": "T014", "origin": "Delhi", "destination": "Varanasi", "distance_km": 820,
     "modes": "Vande Bharat Express, Flight, Superfast Train",
     "travel_time": "Vande Bharat: 8h | Flight: 1h 20m | Overnight Train: 11-13h",
     "estimated_cost_range": "Vande Bharat: ₹1750-₹3300 | Flight: ₹2800-₹6000 | Train: ₹450-₹1800",
     "general_route_info": "New Delhi to Varanasi Vande Bharat is one of the most reliable and fastest overland travel options."}
]

df_trans = pd.DataFrame(transport_data)
df_trans.to_csv("data/raw/transport_raw.csv", index=False)
print("Saved data/raw/transport_raw.csv with", len(df_trans), "records")

# 6. Packing Tips Raw
packing_tips_data = [
    {
        "tip_id": "P001",
        "category": "Monsoon Travel",
        "packing_items": "Sturdy windproof umbrella, waterproof raincoat / poncho, waterproof backpack cover, silica gel sachets for electronics, quick-dry microfibre towels, non-slip waterproof trekking sandals / gumboots, mosquito repellent cream (Odomos), waterproof zip-lock pouches for mobile & wallet.",
        "clothing": "Synthetic / nylon quick-drying shirts and shorts, dark-colored trousers (mud-resistant), extra pairs of cotton socks, avoid heavy denim which stays wet for hours.",
        "health_safety": "Carry anti-fungal powder, water purification tablets, band-aids, ORS hydration salts, anti-diarrhea tablets. Avoid stepping near rushing edge currents or unbarricaded waterfall rims.",
        "travel_advice": "Check state meteorological department (IMD) red/orange rain alerts before heading to Western Ghats. Landslide-prone mountain roads often face temporary traffic diversions."
    },
    {
        "tip_id": "P002",
        "category": "Winter / Hill Station",
        "packing_items": "Thermal innerwear (upper and lower base layers), fleece jacket, windproof heavy down feather jacket, woolen beanie / monkey cap, woolen neck muffler, insulated touchscreen gloves, lip balm, moisturizing body cold cream, sunglasses with UV protection for snow glare.",
        "clothing": "3-layer clothing strategy: moisture-wicking base thermal, insulating fleece/wool mid-layer, waterproof/windproof outer shell. Woolen socks and sturdy boots.",
        "health_safety": "Cold remedy medicines, throat lozenges, vaporub, thermocool pads, hand warmers, prescription medicines for high altitude (Diamox if visiting high Himalayas above 10,000 ft).",
        "travel_advice": "Days are shorter in the mountains during winter; start sightseeing early in the morning and reach hotel before post-sunset freezing temperatures set in."
    },
    {
        "tip_id": "P003",
        "category": "Beach / Coastal",
        "packing_items": "Broad-spectrum reef-safe sunscreen (SPF 50+), UV polarized sunglasses, wide-brimmed sun hat, beach sarong / mat, dry bag for boat rides, waterproof phone pouch, flip flops, aloe vera gel for sunburn relief, reusable water bottle.",
        "clothing": "Breathable lightweight cotton or linen shirts, sundresses, swimsuits, rash guards for water sports, board shorts, comfortable walking sandals.",
        "health_safety": "Electrolyte packets, motion sickness tablets (Avomine) if planning boat or ferry cruises, calamine lotion for jellyfish or sandfly stings, stay hydrated with coconut water.",
        "travel_advice": "Respect red flags placed by lifeguards on beaches. Never swim under the influence of alcohol or during rough high tide conditions."
    },
    {
        "tip_id": "P004",
        "category": "Desert / Rajasthan",
        "packing_items": "Cotton scarf / shemagh to shield mouth and hair from blowing sand, high SPF sunscreen, polarized sunglasses, dust-proof camera cover, power bank (phone batteries drain faster in extremes), wet wipes, lip balm with SPF.",
        "clothing": "Breathable, loose light-colored full-sleeve cotton clothes for blazing daytime sun. Warm fleece jacket or shawl for cold desert night temperatures (desert experiences sharp diurnal temperature drops).",
        "health_safety": "Continuous hydration with electrolytes, glucose powder, eye drops to flush out desert sand, moisturizer.",
        "travel_advice": "Early morning and late afternoon are the best times for fort and desert explorations. Avoid open sightseeing during 12 PM - 3 PM peak heat."
    },
    {
        "tip_id": "P005",
        "category": "Spiritual / Heritage",
        "packing_items": "Slip-on footwear that is easy to remove outside temples, a clean cloth tote bag to store shoes, modest shawl or scarf to cover head and shoulders, hand sanitizer, small cash denominations for temple offerings.",
        "clothing": "Modest traditional or semi-formal clothing covering knees and shoulders (kurta, churidar, long trousers, sarees). Avoid sleeveless tops or short shorts in sanctum sanctorum.",
        "health_safety": "Socks if walking on temple stone floors heated by sun or cold marble at dawn. Drink bottled or filtered water.",
        "travel_advice": "Respect photography bans inside ancient shrines. Beware of touts claiming special fast-track blessings for exorbitant sums."
    },
    {
        "tip_id": "P006",
        "category": "Trekking / Adventure",
        "packing_items": "Ankle-support trekking boots with deep lug grip, trekking poles for descent balance, 30L-40L ergonomic backpack, LED headlamp with extra batteries, Swiss army pocket knife, light rain jacket, energy bars, trail mix nuts, personal first aid kit.",
        "clothing": "Moisture-wicking dry-fit polyester t-shirts, convertible cargo trekking pants, anti-blister technical socks.",
        "health_safety": "Crepe bandage, antiseptic spray (Betadine), muscle spray (Volini), blister plasters, glucose-D, pain relievers.",
        "travel_advice": "Never trek alone on unmapped trails. Inform your base camp or lodge host about your planned route and estimated return time."
    },
    {
        "tip_id": "P007",
        "category": "Solo Travel Safety",
        "packing_items": "Emergency whistle, small door stopper / portable door lock for hotel room security, secondary hidden wallet with emergency cash & backup debit card, pepper spray (check transport regulations), fully charged 20000mAh power bank, printed photocopies of ID proofs.",
        "clothing": "Comfortable inconspicuous casual attire suited to local regional customs.",
        "health_safety": "Share live location on WhatsApp / Google Maps with a trusted family member or friend. Save local emergency police (112) and women helpline numbers.",
        "travel_advice": "Avoid arriving in unknown transit stations late at night without pre-booked transport. Trust your gut instinct; if a situation feels unsafe, step into a busy public store or restaurant immediately."
    }
]

df_pack = pd.DataFrame(packing_tips_data)
df_pack.to_csv("data/raw/packing_tips_raw.csv", index=False)
print("Saved data/raw/packing_tips_raw.csv with", len(df_pack), "records")
