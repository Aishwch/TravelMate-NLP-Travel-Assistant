# Project Scope & Limitations — TravelMate

## 1. Project In-Scope Capabilities

### Geographic & Domain Coverage
* **Target Geographic Region**: Major Indian travel circuits spanning 36 prominent destinations across 17 states and union territories (Western Ghats, Rajasthan Heritage, Goa Coastal, Himalayan Hill Stations, South Indian Backwaters & Coffee Plantations, North-East scenic corridors, and Spiritual pilgrim cities).
* **Domain Elements**:
  * Destination discovery and multi-attribute recommendation.
  * Single-city and multi-day itinerary scheduling (1 to 7 days).
  * Travel expenditure estimation across Budget, Mid-range, and Luxury tiers.
  * Regional culinary guides and iconic dining spots with dietary filtering (Veg, Non-Veg).
  * Inter-city transportation modes (Flight, Train, Bus, Highway drive) and travel durations.
  * Context-aware packing checklists and travel safety guidelines (Monsoon, High Altitude Winter, Coastal, Desert, Solo Travel).
  * Side-by-side comparative analysis of two destinations.

### NLP & Technical Scope
* **Natural Language Queries**: Supports open-ended phrasing, informal slang, contractions, shorthand budget notations (*"8k"*, *"10000 rs"*), and common spelling errors (*"Mumbay"*, *"Mahabaleshwr"*).
* **Multi-Turn Dialogues**: Tracks session state across consecutive conversational turns and resolves elliptical follow-up questions.
* **Offline Local Execution**: Operates entirely on the local machine without requiring paid external third-party API keys (e.g. OpenAI, Google Gemini, paid flight aggregators).

---

## 2. Out-of-Scope Elements & Limitations

1. **Real-Time Dynamic Pricing**:
   * Hotel tariffs, airline ticket prices, and train seat availability fluctuate continuously based on dynamic demand algorithms. TravelMate provides verified historical benchmarks rather than fabricating live seat availability.
2. **Live Booking & Transactional Payment Processing**:
   * The project functions as an intelligent advisory and planning system; it does not process live payment gateway transactions or issue PNR booking confirmations.
3. **International Tourism Beyond India**:
   * The current knowledge base is focused on Indian tourism destinations. Expanding to international hubs (Europe, Southeast Asia) is architecturally supported by simply appending records to `data/processed/destinations.csv`.
4. **Real-Time Weather Feeds**:
   * Weather advisories are mapped using historical seasonal patterns (Monsoon: Jun-Sep, Winter: Nov-Feb, Summer: Mar-May). Live hourly precipitation radar requires optional third-party weather API keys documented in `.env.example`.
