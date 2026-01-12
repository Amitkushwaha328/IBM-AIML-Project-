import streamlit as st
import pandas as pd
import math
import os
import time
import random
import google.generativeai as genai
from fpdf import FPDF

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="TravelGenie Pro",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. UI STYLING (Dark Mode)
# ==========================================
st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at center, #0f172a 0%, #000000 100%); color: white; }
    h1, h2, h3 { color: #f8fafc !important; }
    p, label, span, div { color: #cbd5e1 !important; }
    .stSelectbox div[data-baseweb="select"] > div, .stTextInput div[data-baseweb="input"] > div {
        background-color: #1e293b !important; color: white !important; border: 1px solid #334155 !important;
    }
    div[data-testid="stMetric"] { background: rgba(30, 41, 59, 0.5); border-radius: 10px; border: 1px solid #334155; }
    div[data-testid="stMetricValue"] { color: #38bdf8 !important; }
    .stButton > button { background: linear-gradient(90deg, #3b82f6, #2563eb); color: white; border: none; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. ROBUST DATA ENGINE
# ==========================================
def safe_read_csv(filename):
    if not os.path.exists(filename):
        return pd.DataFrame()
    try:
        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip().str.lower()
        rename_map = {
            'location': 'city', 'place_name': 'name', 'place': 'name',
            'hotel_name': 'name', 'hotel_price': 'price', 'cost': 'price',
            'description': 'desc', 'about': 'desc', 'city_desc': 'desc',
            'province': 'state', 'territory': 'state',
            'best_time_to_visit': 'best_time'
        }
        df.rename(columns=rename_map, inplace=True)
        df = df.loc[:, ~df.columns.duplicated()]
        return df
    except Exception as e:
        st.error(f"Error reading {filename}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_data():
    return {
        "Cities": safe_read_csv("clean_city.csv"),
        "Hotels": safe_read_csv("clean_hotel.csv"),
        "Places": safe_read_csv("clean_places.csv"),
        "Food": safe_read_csv("clean_food.csv"),
        "Transport": safe_read_csv("clean_transport.csv"),
    }

data = load_data()

def parse_price(val):
    try:
        if pd.isna(val): return 0.0
        return float(str(val).replace('₹', '').replace(',', '').strip())
    except:
        return 0.0

# PDF Generator Helper
def create_pdf(city, days, travelers, total_cost, itinerary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=f"Trip Plan: {city}", ln=1, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Duration: {days} Days | Travelers: {travelers} | Est. Cost: {int(total_cost)} INR", ln=1, align='C')
    pdf.ln(10)
    
    # Clean text for PDF
    clean_text = itinerary_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=clean_text)
    return bytes(pdf.output(dest="S"))

# ==========================================
# 4. SIDEBAR & SESSION STATE
# ==========================================
if 'selected_city' not in st.session_state:
    st.session_state['selected_city'] = None
if 'run_calc' not in st.session_state:
    st.session_state['run_calc'] = False

st.sidebar.title("⚡ TravelGenie Pro")
mode = st.sidebar.radio("Mode", ["🌍 Trip Planner", "🔐 Admin Dashboard"])

st.sidebar.markdown("---")

# --- API KEY HANDLING  ---
api_key = st.secrets.get("GEMINI_API_KEY", None)

if api_key:
    st.sidebar.markdown(
    """
    <div style="
        background: rgba(34,197,94,0.15);
        padding:10px;
        border-radius:8px;
        color:#22c55e;
        font-weight:600;
        text-align:center;
    ">
        🤖 AI Ready (Gemini Connected)
    </div>
    """,
    unsafe_allow_html=True
)

else:
    st.sidebar.warning("⚠️ Gemini API not configured")
    api_key = st.sidebar.text_input(
        "🔑 Google Gemini API Key",
        type="password",
        placeholder="Paste API key"
    )

# ==========================================
# 5. TRIP PLANNER LOGIC
# ==========================================
if mode == "🌍 Trip Planner":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Configuration")
    
    # 1. Origin
    origins = ["Delhi", "Mumbai", "Ahmedabad", "Bihar"]
    if not data['Transport'].empty and 'from_state' in data['Transport']:
        origins = sorted(data['Transport']['from_state'].dropna().unique())
    origin = st.sidebar.selectbox("Starting From", origins)
    
    # 2. Method
    method = st.sidebar.radio("Selection Method", ["AI Matcher", "Manual"], horizontal=True)

    # ===== MODE INDICATOR IN MAIN SCREEN ===== #
    st.markdown("<br>", unsafe_allow_html=True)

    if method == "AI Matcher":
        st.markdown(
        """
            <div style="text-align:center; padding:80px 20px;">
                <h2>🤖 AI Matcher Mode</h2>
                <p style="font-size:18px; color:#94a3b8;">
                    Describe your vibe and let AI choose the destination.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="text-align:center; padding:80px 20px;">
                <h2>🧭 Manual Selection Mode</h2>
                <p style="font-size:18px; color:#94a3b8;">
                    Select state and destination manually.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    
    if method == "AI Matcher":
        vibe = st.sidebar.text_area("Describe your trip (e.g. peaceful snow, temple tour, adventure):")
        
        if st.sidebar.button("🔮 Find Destination"):
            # A. TRY GEMINI AI FIRST
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-flash-latest')

                    
                    city_list = ", ".join(data['Cities']['city'].unique()) if not data['Cities'].empty else "Manali, Goa, Jaipur, Kerala"
                    
                    prompt = f"""
                    I have a user who wants this vibe: "{vibe}".
                    Pick the SINGLE best city from this list: [{city_list}].
                    Return ONLY the city name. Nothing else.
                    """
                    response = model.generate_content(prompt)
                    suggested_city = response.text.strip()
                    
                    st.session_state['selected_city'] = suggested_city
                    st.session_state['run_calc'] = True
                    st.sidebar.success(f"AI Suggests: {suggested_city}")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"AI Error: {e}")
            
            # B. FALLBACK TO KEYWORD MATCHING
            else:
                if not data['Cities'].empty:
                    best_city = None; best_score = -1
                    tokens = vibe.lower().split()
                    for _, row in data['Cities'].iterrows():
                        score = 0
                        txt = str(row.get('desc', '')) + " " + str(row.get('best_time', '')) + " " + str(row.get('city', ''))
                        txt = txt.lower()
                        for t in tokens:
                            if t in txt: score += 1
                        if score > best_score:
                            best_score = score
                            best_city = row.get('city')
                    
                    if best_city:
                        st.session_state['selected_city'] = best_city
                        st.session_state['run_calc'] = True
                        st.sidebar.success(f"Recommended: {best_city}")
                        st.rerun()
                    else:
                        st.sidebar.warning("No keyword match found. Try 'Snow' or 'Beach'.")

    else:
        # Manual Selection
        states = ["All"]
        if not data['Cities'].empty and 'state' in data['Cities']:
            states = ["All"] + sorted(data['Cities']['state'].dropna().unique())
        
        f_state = st.sidebar.selectbox("Filter State", states)
        
        city_opts = sorted(data['Cities']['city'].unique()) if not data['Cities'].empty else []
        if f_state != "All" and not data['Cities'].empty:
            city_opts = sorted(data['Cities'][data['Cities']['state'] == f_state]['city'].unique())
            
        # Smart Indexing
        idx = 0
        if st.session_state['selected_city'] in city_opts:
            idx = city_opts.index(st.session_state['selected_city'])
            
        target_city = st.sidebar.selectbox("Destination", city_opts, index=idx)
        
        if st.sidebar.button("Generate Plan"):
            st.session_state['selected_city'] = target_city
            st.session_state['run_calc'] = True
            st.rerun()

    # Budget Inputs
    st.sidebar.markdown("---")
    days = st.sidebar.slider("Days", 2, 10, 4)
    travelers = st.sidebar.slider("Travelers", 1, 10, 2)
    budget = st.sidebar.number_input("Budget (₹)", 5000, 1000000, 30000)

    # ==========================================
    # 6. RESULTS DISPLAY
    # ==========================================
    if st.session_state['run_calc'] and st.session_state['selected_city']:
        city = st.session_state['selected_city']
        
        try:
            # --- ROBUST CALCULATIONS ---
            
            # 1. Transport
            t_cost = 0; t_name = "Flight/Train"
            if not data['Transport'].empty and 'to_state' in data['Transport']:
                dest_rows = data['Cities'][data['Cities']['city'] == city]
                if not dest_rows.empty:
                    d_state = dest_rows.iloc[0].get('state')
                    route = data['Transport'][
                        (data['Transport']['from_state'] == origin) & 
                        (data['Transport']['to_state'] == d_state)
                    ]
                    if not route.empty:
                        t_cost = parse_price(route.iloc[0].get('price')) * travelers * 2
                        t_name = route.iloc[0].get('train_name', 'Transport')
                    else:
                        t_cost = 3000 * travelers * 2
                else:
                    t_cost = 3000 * travelers * 2
            else:
                 t_cost = 3000 * travelers * 2

            # 2. Hotel
            h_cost = 0; h_name = "City Hotel"
            if not data['Hotels'].empty and 'city' in data['Hotels']:
                hotels = data['Hotels'][data['Hotels']['city'] == city]
                if not hotels.empty:
                    if 'rating' in hotels:
                        best = hotels.sort_values('rating', ascending=False).iloc[0]
                    else:
                        best = hotels.iloc[0]
                    h_name = best.get('name', 'Hotel')
                    h_cost = parse_price(best.get('price', 2000)) * math.ceil(travelers/2) * days
                else:
                    h_cost = 2500 * math.ceil(travelers/2) * days
            else:
                h_cost = 2500 * math.ceil(travelers/2) * days
            
            # 3. Total
            misc = 1000 * travelers * days
            total = t_cost + h_cost + misc
            save = budget - total

            # --- DISPLAY ---
            st.title(f"Trip to {city}")
            st.caption(f"{days} Days | {travelers} Travelers | From {origin}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Cost", f"₹{int(total):,}")
            c2.metric("Transport", f"₹{int(t_cost):,}", t_name)
            c3.metric("Hotel", f"₹{int(h_cost):,}", h_name)
            c4.metric("Savings", f"₹{int(save):,}", delta="Svgs" if save > 0 else "Low")
            
            st.markdown("---")
            
            t1, t2, t3, t4 = st.tabs(["📅 Itinerary", "📍 Places", "🏨 Details", "🗺️ Map"])
            
            # Tab 1: Itinerary
            itinerary_text = ""
            with t1:
                st.subheader(f"{days}-Day Plan")
                
                # Check for AI API first for Itinerary
                if api_key:
                    with st.spinner("AI is crafting your itinerary..."):
                        try:
                            genai.configure(api_key=api_key)
                            # UPDATED MODEL TO FLASH
                            model = genai.GenerativeModel('gemini-1.5-flash')
                            prompt = f"Create a short {days}-day itinerary for {city} for {travelers} people focused on {vibe if 'vibe' in locals() else 'sightseeing'}."
                            response = model.generate_content(prompt)
                            itinerary_text = response.text
                            st.markdown(itinerary_text)
                        except:
                            st.warning("AI failed, showing standard plan.")
                
                # Standard Fallback Logic
                if not itinerary_text:
                    if not data['Places'].empty and 'city' in data['Places']:
                        places_df = data['Places'][data['Places']['city'] == city]
                        all_places = []
                        if not places_df.empty:
                            all_places = places_df['name'].astype(str).values.flatten().tolist()
                        
                        buffer = ""
                        for d in range(1, days+1):
                            buffer += f"**Day {d}**\n"
                            if d==1: buffer += "✈️ Arrival & Check-in\n"
                            elif d==days: buffer += "🛍️ Shopping & Departure\n"
                            else:
                                if len(all_places) > 0:
                                    daily = random.sample(all_places, min(2, len(all_places)))
                                    buffer += f"🔭 Visit: {', '.join(daily)}\n"
                                else:
                                    buffer += "🔭 City Exploration\n"
                            buffer += "\n"
                        st.markdown(buffer)
                        itinerary_text = buffer
                
                # PDF DOWNLOAD BUTTON
                if itinerary_text:
                    pdf_data = create_pdf(city, days, travelers, total, itinerary_text)
                    st.download_button(
                        label="📄 Download Itinerary (PDF)",
                        data=pdf_data,
                        file_name=f"Trip_to_{city}.pdf",
                        mime="application/pdf"
                    )
            
            # Tab 2: Places List
            with t2:
                if not data['Places'].empty and 'city' in data['Places']:
                    places_df = data['Places'][data['Places']['city'] == city]
                    if not places_df.empty:
                        for _, row in places_df.iterrows():
                            n = row.get('name', 'Place')
                            d = row.get('desc', 'Explore this beautiful location.')
                            if pd.isna(d): d = "Explore this beautiful location."
                            r = row.get('rating', '4.5')
                            with st.expander(f"🚩 {n}"):
                                st.write(d)
                                st.caption(f"Rating: {r} ⭐")
                    else:
                        st.write("No specific places found in database.")

            # Tab 3: Hotel & Food
            with t3:
                c1, c2 = st.columns(2)
                with c1: 
                    st.info(f"**Stay:** {h_name}\n\nPrice: ₹{int(h_cost/days)}/night")
                with c2:
                    st.success("**Food:**")
                    if not data['Food'].empty and 'city' in data['Food']:
                        food_df = data['Food'][data['Food']['city'] == city]
                        if not food_df.empty:
                            st.dataframe(food_df[['name', 'price', 'type']], hide_index=True)
                        else:
                            st.write("Try local street food!")

            # Tab 4: Map
            with t4:
                COORDS = {
                    "Manali": {"lat": 32.2432, "lon": 77.1892}, "Goa": {"lat": 15.2993, "lon": 74.1240},
                    "Mumbai": {"lat": 19.0760, "lon": 72.8777}, "Delhi": {"lat": 28.7041, "lon": 77.1025},
                    "Bangalore": {"lat": 12.9716, "lon": 77.5946}, "Srinagar": {"lat": 34.0837, "lon": 74.7973},
                    "Jaipur": {"lat": 26.9124, "lon": 75.7873}, "Udaipur": {"lat": 24.5854, "lon": 73.7125},
                    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714}
                }
                if city in COORDS:
                    st.map(pd.DataFrame([COORDS[city]]), zoom=10)
                else:
                    st.warning("Map coordinates not available for this city.")

        except Exception as e:
            st.error(f"Calculation Error: {e}")
            st.write("Please check your CSV files contain data for the selected city.")

    else:
        st.markdown("<br><br><h2 style='text-align:center'>✈️ Ready to Travel?</h2>", unsafe_allow_html=True)

# ==========================================
# 7. ADMIN DASHBOARD
# ==========================================
elif mode == "🔐 Admin Dashboard":
    st.title("Admin Panel")
    if st.sidebar.text_input("Password", type="password") == "admin123":
        tab1, tab2 = st.tabs(["View Data", "Edit Data"])
        
        with tab1:
            sel = st.selectbox("Select Dataset", list(data.keys()))
            st.dataframe(data[sel], use_container_width=True)
            
        with tab2:
            st.warning("Live Editing")
            sel_edit = st.selectbox("Edit Dataset", list(data.keys()), key='edit')
            if not data[sel_edit].empty:
                df_edit = st.data_editor(data[sel_edit], num_rows="dynamic")
                if st.button("Save Changes"):
                    fname = f"clean_{sel_edit.lower().rstrip('s') if sel_edit != 'Cities' else 'city'}.csv"
                    if sel_edit == "Transport": fname = "clean_transport.csv"
                    if sel_edit == "Cities": fname = "clean_city.csv"
                    if sel_edit == "Hotels": fname = "clean_hotel.csv"
                    if sel_edit == "Places": fname = "clean_places.csv"
                    if sel_edit == "Food": fname = "clean_food.csv"
                    
                    df_edit.to_csv(fname, index=False)
                    st.success("Saved!")
                    time.sleep(1)
                    st.rerun()