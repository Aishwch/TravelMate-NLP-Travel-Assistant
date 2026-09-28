"""
TravelMate — Intelligent NLP Travel Assistant
=============================================
Streamlit Web Application
Final Year AI & Data Science Engineering NLP Project
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.chatbot import TravelMateAssistant
from src.preprocessing import preprocess_query
from src.entity_extractor import TravelEntityExtractor
from src.intent_classifier import IntentClassifier
from src.preference_extractor import PreferenceExtractor
from src.semantic_search import SemanticSearchEngine
from src.itinerary import ItineraryGenerator
from src.budget import BudgetPlanner

# Streamlit Page Configuration
st.set_page_config(
    page_title="TravelMate — Intelligent NLP Travel Assistant",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and clean typography
st.markdown("""
<style>
    /* Global styles */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .highlight-card {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #BFDBFE;
        margin-bottom: 1rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
    .badge-green { background-color: #D1FAE5; color: #065F46; }
    .badge-purple { background-color: #EDE9FE; color: #5B21B6; }
    .badge-orange { background-color: #FFEDD5; color: #9A3412; }
    .badge-red { background-color: #FEE2E2; color: #991B1B; }
    
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 0.8rem;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Cached resource initialization
@st.cache_resource
def get_assistant():
    """Instantiates the conversational assistant once and keeps it warm in memory."""
    return TravelMateAssistant()


assistant = get_assistant()

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "👋 **Hello! I'm TravelMate, your Intelligent NLP Travel Assistant.**\n\n"
                       "I understand natural travel queries, budgets, multi-day plans, transport, foods, and preferences.\n\n"
                       "💡 *Try asking:*\n"
                       "* *'I have 3 days and ₹8000 and want a peaceful trip from Mumbai.'*\n"
                       "* *'Plan a 3 day trip to Jaipur.'*\n"
                       "* *'Travelling with my parents and they can't walk too much. Where should we go?'*\n"
                       "* *'Which is better for a weekend, Lonavala or Matheran?'*\n"
                       "* *'What food should I try in Hyderabad?'*"
        }
    ]

# Sidebar Navigation
st.sidebar.image("https://images.unsplash.com/photo-1488646953014-85cb44e25828?w=500&auto=format&fit=crop&q=60", use_container_width=True)
st.sidebar.markdown("## 🌍 TravelMate")
st.sidebar.markdown("*NLP-Based Intelligent Travel Assistance System*")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home",
        "💬 Travel Assistant",
        "🗺️ Explore Destinations",
        "📅 Itinerary Planner",
        "💰 Budget Planner",
        "🔍 NLP Analysis",
        "📊 Model Performance",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("🎓 **Academic Mini Project**  \nArtificial Intelligence & Data Science  \nPowered by spaCy, Scikit-Learn, & Neural Embeddings")


# ==============================================================================
# PAGE 1: 🏠 HOME
# ==============================================================================
if page == "🏠 Home":
    st.markdown('<div class="main-title">🌍 TravelMate</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">An NLP-Based Intelligent Travel Assistance System</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        Welcome to **TravelMate**, an advanced intelligent travel assistant built using modern 
        **Natural Language Processing (NLP)**, **Machine Learning**, and **Semantic Information Retrieval**.
        
        Unlike rigid, rule-based FAQ chatbots with predefined buttons, TravelMate **understands natural language queries**, 
        extracts multidimensional constraints (destinations, budgets, durations, travel party, terrain accessibility, crowd preferences), 
        and computes dynamic recommendations with transparent explainability.
        """)

        st.markdown("### 💡 Quick-Test Sample Queries")
        st.markdown("Click any sample query to start a conversation in the **💬 Travel Assistant**:")

        sample_prompts = [
            "I have 3 days and ₹8000 and want a peaceful trip from Mumbai.",
            "I'm travelling with my parents and they can't walk too much. Where should we go?",
            "Plan a 3 day trip to Jaipur with heritage attractions and food.",
            "Which is better for a weekend, Lonavala or Matheran?",
            "What food should I try in Hyderabad?",
            "What should I pack for a monsoon trip?"
        ]

        for prompt in sample_prompts:
            if st.button(f"👉 \"{prompt}\"", key=prompt):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                res = assistant.respond(prompt)
                st.session_state.chat_history.append({"role": "assistant", "content": res["response"]})
                st.rerun()

    with col2:
        st.markdown('<div class="highlight-card">', unsafe_allow_html=True)
        st.markdown("#### ⚡ System Architecture Highlights")
        st.markdown("""
        * **Hybrid Entity Extractor**: spaCy NER + Regex + Levenshtein Fuzzy String Matching
        * **Intent Classifier**: TF-IDF + Logistic Regression across 26 semantic travel intents
        * **Semantic Search**: Neural Sentence Transformers (`all-MiniLM-L6-v2`) with Cosine Similarity
        * **Dynamic Recommender**: Multi-factor scoring across 8 weighted dimensions with explainability
        * **Conversational Memory**: Session-level context resolution for multi-turn dialogues
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="Verified Destinations", value=len(assistant.destinations_df))
            st.metric(label="Tourist Attractions", value=len(assistant.attractions_df))
        with m_col2:
            st.metric(label="Cuisine Specialties", value=len(assistant.food_df))
            st.metric(label="Semantic Intents", value="26 Categories")


# ==============================================================================
# PAGE 2: 💬 TRAVEL ASSISTANT
# ==============================================================================
elif page == "💬 Travel Assistant":
    st.markdown('<div class="main-title">💬 Travel Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Chat naturally with TravelMate using open-ended questions</div>', unsafe_allow_html=True)

    # Conversational Context Expander (Session Memory Inspector)
    ctx_summary = assistant.context.get_summary()
    with st.expander("🧠 Active Conversational Context (Session Memory Inspector)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.write(f"**Active Destination**: `{ctx_summary.get('active_destination') or 'None'}`")
        c2.write(f"**Duration**: `{str(ctx_summary.get('duration_days')) + ' days' if ctx_summary.get('duration_days') else 'None'}`")
        c3.write(f"**Budget**: `{('₹' + str(ctx_summary.get('budget_inr'))) if ctx_summary.get('budget_inr') else 'None'}`")
        c4.write(f"**Origin**: `{ctx_summary.get('origin') or 'None'}`")
        
        pref_str = ", ".join(ctx_summary.get("preferences", [])) or "None"
        st.write(f"**Active Preferences**: `{pref_str}` | **Party Size**: `{ctx_summary.get('people_count') or 1}` | **Turns**: `{ctx_summary.get('turn_count')}`")
        
        if st.button("🗑️ Reset Conversational Memory"):
            assistant.context.reset()
            st.session_state.chat_history = [
                {"role": "assistant", "content": "Conversational memory reset. How can I help you plan your travels?"}
            ]
            st.rerun()

    # Render Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input Box
    user_input = st.chat_input("Ask anything travel-related (e.g. destinations, itineraries, budgets, food, packing)...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Analyzing query with NLP pipeline..."):
            response_dict = assistant.respond(user_input)
            response_text = response_dict["response"]

        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)


# ==============================================================================
# PAGE 3: 🗺️ EXPLORE DESTINATIONS
# ==============================================================================
elif page == "🗺️ Explore Destinations":
    st.markdown('<div class="main-title">🗺️ Explore Destinations</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Browse verified destinations across Indian travel circuits</div>', unsafe_allow_html=True)

    df = assistant.destinations_df.copy()

    # Search & Filter Controls
    f_col1, f_col2, f_col3, f_col4 = st.columns([2, 1, 1, 1])
    search_query = f_col1.text_input("🔍 Search destination or landmark", "")
    all_states = ["All States"] + sorted(list(df["state"].dropna().unique()))
    selected_state = f_col2.selectbox("State / Region", all_states)
    selected_crowd = f_col3.selectbox("Crowd Level", ["All", "Low", "Moderate", "High"])
    selected_walking = f_col4.selectbox("Walking Requirement", ["All", "Low", "Moderate", "High"])

    # Apply Filters
    if search_query:
        mask = (
            df["destination"].str.contains(search_query, case=False, na=False) |
            df["description"].str.contains(search_query, case=False, na=False) |
            df["famous_for"].str.contains(search_query, case=False, na=False)
        )
        df = df[mask]

    if selected_state != "All States":
        df = df[df["state"] == selected_state]

    if selected_crowd != "All":
        df = df[df["crowd_level"] == selected_crowd]

    if selected_walking != "All":
        df = df[df["walking_level"] == selected_walking]

    st.write(f"Showing **{len(df)}** destinations:")

    # Cards Display
    for _, row in df.iterrows():
        with st.container():
            st.markdown(f"### 📍 {row['destination']}, {row['state']}")
            st.write(f"{row['description']}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**⭐ Rating**: `{row['rating']}/5.0`")
            c2.markdown(f"**💰 Est. Daily**: `₹{row['estimated_cost_per_day']:,}/day`")
            c3.markdown(f"**👥 Crowd Density**: `{row['crowd_level']}`")
            c4.markdown(f"**🚶 Walking Exertion**: `{row['walking_level']}`")
            
            st.markdown(f"**🏷️ Famous For**: {row['famous_for']}")
            st.markdown(f"**🏛️ Top Attractions**: {row['attractions']}")
            st.markdown("---")


# ==============================================================================
# PAGE 4: 📅 ITINERARY PLANNER
# ==============================================================================
elif page == "📅 Itinerary Planner":
    st.markdown('<div class="main-title">📅 Dynamic Itinerary Planner</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Generate realistic day-by-day schedules organized by visit duration & proximity</div>', unsafe_allow_html=True)

    dest_list = sorted(list(assistant.destinations_df["destination"].unique()))
    
    col1, col2, col3 = st.columns([2, 1, 1])
    sel_dest = col1.selectbox("Select Target Destination", dest_list, index=0)
    sel_days = col2.slider("Trip Duration (Days)", min_value=1, max_value=7, value=3)
    sel_party = col3.selectbox("Travel Group", ["Standard", "Solo", "Friends", "Couple", "Parents / Family"])

    st.write("**Preferred Interests (Optional):**")
    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    pref_nature = p_col1.checkbox("Nature / Scenic")
    pref_history = p_col2.checkbox("Heritage / Forts")
    pref_peace = p_col3.checkbox("Peaceful / Calm")
    pref_food = p_col4.checkbox("Food & Dining")
    pref_adv = p_col5.checkbox("Adventure")

    active_prefs = []
    if pref_nature: active_prefs.append("nature")
    if pref_history: active_prefs.append("historical")
    if pref_peace: active_prefs.append("peaceful")
    if pref_food: active_prefs.append("food")
    if pref_adv: active_prefs.append("adventure")

    if st.button("✨ Generate Custom Itinerary", type="primary"):
        with st.spinner("Generating itinerary from structured attractions knowledge base..."):
            itin_data = assistant.itinerary_gen.generate_itinerary(
                destination=sel_dest,
                days=sel_days,
                preferences=active_prefs,
                travel_group=sel_party
            )
            formatted_text = assistant.itinerary_gen.format_as_text(itin_data)
            st.markdown(formatted_text)
            st.download_button(
                label="📥 Download Itinerary (Markdown)",
                data=formatted_text,
                file_name=f"{sel_dest}_{sel_days}day_itinerary.md",
                mime="text/markdown"
            )


# ==============================================================================
# PAGE 5: 💰 BUDGET PLANNER
# ==============================================================================
elif page == "💰 Budget Planner":
    st.markdown('<div class="main-title">💰 Intelligent Budget Planner</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Estimate realistic accommodation, dining, transit, and activity expenditures</div>', unsafe_allow_html=True)

    dest_list = sorted(list(assistant.budget_df["destination"].unique()))

    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    dest_choice = b_col1.selectbox("Destination", dest_list)
    days_choice = b_col2.slider("Duration (Days)", min_value=1, max_value=14, value=3)
    people_choice = b_col3.slider("Number of Travelers", min_value=1, max_value=10, value=2)
    budget_tier = b_col4.selectbox("Budget Tier", ["Mid-range", "Budget", "Luxury"])

    user_budget_input = st.number_input(
        "Enter Your Target Budget in INR (Optional for feasibility check)",
        min_value=0,
        value=15000,
        step=1000
    )

    if st.button("📊 Calculate Budget Breakdown", type="primary"):
        plan = assistant.budget_planner.plan_budget(
            destination=dest_choice,
            total_budget=user_budget_input if user_budget_input > 0 else None,
            duration_days=days_choice,
            people_count=people_choice,
            tier=budget_tier
        )

        st.markdown(f"### 💳 Total Estimated Cost: **₹{plan['total_estimated']:,}**")
        st.write(f"*Average: ₹{plan['per_person_per_day']:,} per person / day*")

        if plan.get("feasibility_message"):
            if plan["feasibility_status"] == "Comfortable":
                st.success(plan["feasibility_message"])
            elif plan["feasibility_status"] == "Shoestring / Manageable":
                st.warning(plan["feasibility_message"])
            else:
                st.error(plan["feasibility_message"])

        # Plotly Donut Chart
        bd = plan["breakdown"]
        labels = ["Accommodation", "Meals & Food", "Local Transit", "Activities & Entry", "Emergency Buffer (10%)"]
        values = [bd["accommodation"], bd["food_dining"], bd["local_transport"], bd["activities_sightseeing"], bd["contingency_buffer"]]

        fig = px.pie(
            names=labels,
            values=values,
            hole=0.45,
            title=f"Expenditure Distribution for {dest_choice} ({days_choice} Days, {people_choice} Persons)",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# PAGE 6: 🔍 NLP ANALYSIS (COLLEGE VIVA SHOWCASE)
# ==============================================================================
elif page == "🔍 NLP Analysis":
    st.markdown('<div class="main-title">🔍 NLP Pipeline & Viva Demonstration</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Live interactive visualization of tokenization, entities, intent classification, and semantic embeddings</div>', unsafe_allow_html=True)

    viva_samples = [
        "I want to visit Goa for 3 days with my friends and I love beaches.",
        "I'm travelling from Mumbai for just two days. I don't want to spend more than ₹7000, I love nature and photography, and I want somewhere that isn't too crowded.",
        "Travelling with my parents and they can't walk too much. Where should we go?",
        "Which is better for a weekend, Lonavala or Matheran?",
        "Plan a 3 day trip to Jaipur",
        "What food should I try in Hyderabad?"
    ]
    preset_choice = st.selectbox("Select a preset demonstration query or enter custom text below:", viva_samples)
    nlp_query = st.text_input("Custom Query for NLP Pipeline Analysis", value=preset_choice)

    if st.button("🔬 Analyze with NLP Pipeline", type="primary"):
        st.markdown("---")

        # Stage 1: Preprocessing & Normalization
        st.markdown("### 1️⃣ Query Preprocessing & Normalization")
        prep = preprocess_query(nlp_query)
        
        c1, c2 = st.columns(2)
        c1.markdown(f"**Original Query:** `{prep['original_query']}`")
        c2.markdown(f"**Normalized Text (for ML):** `{prep['normalized_text']}`")

        # Tokenization & Lemmatization Table
        token_data = []
        for t, l, s in zip(prep["tokens_without_stopwords"], prep["lemmas"], prep["stems"]):
            token_data.append({"Token": t, "Lemma (Base Form)": l, "Porter Stem": s})
        
        st.dataframe(pd.DataFrame(token_data), use_container_width=True)

        # Stage 2: Hybrid Entity Extraction
        st.markdown("### 2️⃣ Hybrid Travel Entity Extraction")
        entities = assistant.entity_extractor.extract_all_entities(nlp_query)
        
        e_col1, e_col2, e_col3, e_col4 = st.columns(4)
        e_col1.markdown(f"**Destination**: `{entities['destination'] or 'None'}`")
        e_col2.markdown(f"**Origin**: `{entities['origin'] or 'None'}`")
        e_col3.markdown(f"**Duration**: `{str(entities['duration_days']) + ' days' if entities['duration_days'] else 'None'}`")
        e_col4.markdown(f"**Budget**: `{('₹' + str(entities['budget_inr'])) if entities['budget_inr'] else 'None'}`")

        e_col5, e_col6, e_col7, e_col8 = st.columns(4)
        e_col5.markdown(f"**Travel Group**: `{entities['travel_group'] or 'None'}`")
        e_col6.markdown(f"**Party Count**: `{entities['people_count'] or 'None'}`")
        e_col7.markdown(f"**Crowd Preference**: `{entities['crowd_preference'] or 'Standard'}`")
        e_col8.markdown(f"**Walking Exertion**: `{entities['walking_preference'] or 'Standard'}`")

        # Stage 3: Intent Classification
        st.markdown("### 3️⃣ Intent Classification (TF-IDF + Logistic Regression)")
        top_intent, conf, top_5 = assistant.intent_classifier.predict_with_confidence(nlp_query, top_k=5)
        
        st.markdown(f"**Predicted Intent:** <span class='metric-badge badge-green'>{top_intent.upper()}</span> | **Confidence:** `{conf * 100:.2f}%`", unsafe_allow_html=True)
        
        # Horizontal Bar Chart for Intent Probabilities
        df_intent = pd.DataFrame(list(top_5.items()), columns=["Intent", "Probability"]).sort_values(by="Probability", ascending=True)
        fig_intent = px.bar(
            df_intent,
            x="Probability",
            y="Intent",
            orientation="h",
            title="Top-5 Intent Probability Distribution",
            color="Probability",
            color_continuous_scale="Blues"
        )
        fig_intent.update_layout(xaxis=dict(range=[0, 1]))
        st.plotly_chart(fig_intent, use_container_width=True)

        # Stage 4: Semantic Preference Extraction
        st.markdown("### 4️⃣ Semantic Travel Preferences")
        prefs = assistant.preference_extractor.extract_preferences(nlp_query)
        if prefs:
            badges_html = " ".join([f"<span class='metric-badge badge-purple'>{p.title()}</span>" for p in prefs])
            st.markdown(badges_html, unsafe_allow_html=True)
        else:
            st.write("No explicit preference keywords detected; default broad recommendation applied.")

        # Stage 5: Semantic Embedding Similarity Matches
        st.markdown("### 5️⃣ Neural Semantic Similarity Matches (all-MiniLM-L6-v2)")
        sem_matches = assistant.semantic_engine.search_destinations(nlp_query, top_k=5)
        
        for dest, sim in sem_matches:
            st.markdown(f"**{dest['destination']}**, {dest.get('state', '')} — *Cosine Similarity:* `{sim:.4f}`")
            st.progress(min(max(float(sim), 0.0), 1.0))


# ==============================================================================
# PAGE 7: 📊 MODEL PERFORMANCE
# ==============================================================================
elif page == "📊 Model Performance":
    st.markdown('<div class="main-title">📊 Intent Classifier Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Actual calculated machine learning evaluation metrics on the unseen test split</div>', unsafe_allow_html=True)

    metrics_file = os.path.join(ROOT_DIR, "models", "intent_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        m2.metric("Weighted Precision", f"{metrics['precision_weighted'] * 100:.2f}%")
        m3.metric("Weighted Recall", f"{metrics['recall_weighted'] * 100:.2f}%")
        m4.metric("Weighted F1-Score", f"{metrics['f1_weighted'] * 100:.2f}%")

        st.markdown("---")
        st.markdown("### 🧮 Confusion Matrix Heatmap")
        
        cm = np.array(metrics["confusion_matrix"])
        classes = metrics["classes"]

        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Intent", y="True Intent", color="Count"),
            x=classes,
            y=classes,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Confusion Matrix (Unseen 20% Stratified Test Split)"
        )
        fig_cm.update_layout(height=650)
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("### 📋 Detailed Classification Report")
        report = metrics["classification_report"]
        report_rows = []
        for cls_name in classes:
            if cls_name in report:
                row = report[cls_name]
                report_rows.append({
                    "Intent Class": cls_name,
                    "Precision": f"{row['precision'] * 100:.2f}%",
                    "Recall": f"{row['recall'] * 100:.2f}%",
                    "F1-Score": f"{row['f1-score'] * 100:.2f}%",
                    "Test Support": row["support"]
                })
        st.dataframe(pd.DataFrame(report_rows), use_container_width=True)
    else:
        st.warning("Model metrics file not found. Run prepare_all.py or train_intent_model.py first.")


# ==============================================================================
# PAGE 8: ℹ️ ABOUT & ACADEMIC DOCUMENTATION
# ==============================================================================
elif page == "ℹ️ About":
    st.markdown('<div class="main-title">ℹ️ About & Academic Documentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Comprehensive engineering project documentation and viva defense guide</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📘 Project Abstract & Architecture", "🎓 Viva Defense Questions & Answers", "📊 Dataset Provenance"])

    with tab1:
        st.markdown("""
        ### Abstract
        **TravelMate** is an intelligent travel assistance system developed as an Artificial Intelligence & Data Science 
        engineering NLP Mini Project. The system enables conversational travel discovery and planning by processing unconstrained 
        natural language queries. Unlike rule-based question-answer bots, TravelMate integrates a multi-stage NLP pipeline:
        
        1. **Query Preprocessing**: Custom tokenization preserving travel currency and duration tokens, selective stopword removal, and lemmatization.
        2. **Hybrid Entity Extraction**: spaCy NER combined with regular expressions and Levenshtein fuzzy string distance for typo-tolerant destination recognition.
        3. **Intent Detection**: TF-IDF vectorization with Logistic Regression classifying user queries into 26 semantic travel intents.
        4. **Semantic Embedding Retrieval**: Dense vector embeddings generated via `sentence-transformers` (`all-MiniLM-L6-v2`) with cosine similarity.
        5. **Dynamic Recommendation Engine**: Multi-criteria weighted relevance scoring combining semantic similarity, budget compatibility, duration fit, crowd density, and accessibility.
        6. **Multi-Turn Context Management**: Session memory resolving pronouns and maintaining conversational context across dialogue turns.
        """)

        st.markdown("""
        ```text
                                USER
                                  │
                                  ▼
                        Natural Language Query
                                  │
                                  ▼
                        NLP Preprocessing & Cleaning
                                  │
                                  ▼
                      ┌──────────────────────────┐
                      │      NLP PIPELINE        │
                      │                          │
                      │ Travel-Aware Tokenizer   │
                      │ Stopword Filter (Travel) │
                      │ spaCy NER & Regex        │
                      │ Levenshtein Fuzzy Match  │
                      │ Intent Classifier (TFIDF)│
                      │ Preference Extractor     │
                      └────────────┬─────────────┘
                                  │
                                  ▼
                     Query & Context Understanding
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              Structured       Semantic       Keyword /
              Filtering        Embeddings     Category Match
                    │             │             │
                    └─────────────┼─────────────┘
                                  ▼
                     Dynamic Recommendation Engine
                                  │
                                  ▼
                        Response Generation
        ```
        """)

    with tab2:
        st.markdown(r"""
        ### 🎓 Essential Viva Questions & Model Answers

        #### Q1: Why use NLP instead of standard web forms or dropdown menus?
        **Answer**: Travelers naturally think in multidimensional natural language queries containing soft and hard constraints simultaneously (e.g., *"I have 3 days, ₹8000, travelling with parents, and want a peaceful place"*). Form fields force artificial separation of these nuances. NLP allows users to express constraints, desires, and travel party limitations in one sentence.

        #### Q2: Why is generic spaCy NER insufficient for travel entity extraction?
        **Answer**: Generic spaCy models (`en_core_web_sm`) are trained on news corpora (OntoNotes 5). They frequently miss Indian regional travel entities, confuse travel durations (*"3 days"*) with generic cardinal numbers, fail to recognize Indian currency notations (*"₹8000"*, *"8k"*), and do not handle misspellings (*"Mumbay"*). TravelMate solves this by implementing a **hybrid extractor** that combines spaCy with specialized regex patterns and Levenshtein fuzzy matching against the verified destination catalog.

        #### Q3: How does Semantic Search differ from simple Keyword Search?
        **Answer**: Keyword search requires exact lexical overlap. If a user asks for *"a relaxing place away from crowds"*, a keyword search fails if the destination description says *"peaceful, tranquil hill retreat"*. Semantic search maps sentences into a dense 384-dimensional vector space using neural sentence embeddings (`all-MiniLM-L6-v2`), where semantically related concepts are close in cosine distance regardless of vocabulary differences.

        #### Q4: How does the Recommendation Engine rank destinations?
        **Answer**: It uses a dynamic composite scoring function:
        $$\text{Final Score} = \frac{\sum w_i \cdot S_i}{\sum w_i}$$
        Where components $S_i$ evaluate semantic similarity, preference overlap, budget compatibility, traveler rating, duration fit, season alignment, and physical accessibility (walking exertion). Weights $w_i$ adapt dynamically based on user constraints (e.g., budget weight increases if a strict budget is given).
        """)

    with tab3:
        st.markdown(f"""
        ### 📊 Real External Datasets Documentation
        All data records are derived from legitimate open data sources documented in `DATA_SOURCES.md`:
        * **Destinations**: Wikidata Tourism Knowledge Graph (CC0) — **{len(assistant.destinations_df)} canonical destinations** across Indian states.
        * **Attractions**: Wikidata Tourism Knowledge Graph (CC0) — **{len(assistant.attractions_df)} tourist attractions** with GPS coordinates and visit durations.
        * **Cuisine**: Kaggle Indian Food 101 Dataset (PDDL/CC0) — **{len(assistant.food_df)} authentic culinary records** with dietary tags (veg/non-veg).
        * **Budget**: Canonical destination cost benchmarks — **{len(assistant.budget_df)} destination cost models** across Budget, Mid-range, and Luxury tiers.
        * **Transport**: MoRTH & Indian Railways via Open Government Data (`data.gov.in`) — **{len(assistant.transport_df)} intercity transit routes**.
        * **Packing & Tips**: Seasonal advisory guidelines — **{len(assistant.packing_df)} climate and travel context zones**.
        """)
