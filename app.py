import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# --------------------------
# Configuration & Setup
# --------------------------

st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "housing_price_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "model_features.pkl"
DATA_PATH = BASE_DIR / "data" / "Housing.csv"

# --------------------------
# Cached Data Loaders
# --------------------------

@st.cache_resource
def load_model_and_features():
    """Load trained model and feature columns"""
    try:
        model = joblib.load(MODEL_PATH)
        feature_cols = joblib.load(FEATURE_PATH)
        return model, feature_cols, None
    except Exception as e:
        return None, None, str(e)


@st.cache_data
def load_data():
    """Load and preprocess housing data"""
    try:
        df = pd.read_csv(DATA_PATH)
        df.columns = df.columns.str.strip()

        rename_map = {
            "Date": "date",
            "number of bedrooms": "number_of_bedrooms",
            "number of bathrooms": "number_of_bathrooms",
            "living area": "living_area",
            "lot area": "lot_area",
            "number of floors": "number_of_floors",
            "waterfront present": "waterfront_present",
            "number of views": "number_of_views",
            "condition of the house": "condition_of_the_house",
            "grade of the house": "grade_of_the_house",
            "Area of the house(excluding basement)": "area_of_the_house(excluding_basement)",
            "Area of the basement": "area_of_the_basement",
            "Built Year": "built_year",
            "Renovation Year": "renovation_year",
            "Postal Code": "postal_code",
            "Lattitude": "lattitude",
            "Longitude": "longitude",
            "Number of schools nearby": "number_of_schools_nearby",
            "Distance from the airport": "distance_from_the_airport",
            "Price": "price",
        }

        df.rename(
            columns={old: new for old, new in rename_map.items() if old in df.columns},
            inplace=True,
        )

        return df, None
    except Exception as e:
        return None, str(e)


def preprocess_input(raw_input: dict, feature_cols):
    """Transform raw user input into model-ready format"""
    try:
        df_input = pd.DataFrame([raw_input])
        df_input = pd.get_dummies(df_input, columns=["postal_code"], drop_first=True)

        for col in feature_cols:
            if col not in df_input.columns:
                df_input[col] = 0

        df_input = df_input[feature_cols]
        return df_input, None
    except Exception as e:
        return None, str(e)


def get_similar_properties(df, user_input, n=5):
    """Find similar properties in the dataset"""
    features = ['number_of_bedrooms', 'number_of_bathrooms', 'living_area', 'grade_of_the_house']
    
    df_copy = df.copy()
    for feat in features:
        if feat in df_copy.columns:
            df_copy[f'{feat}_diff'] = abs(df_copy[feat] - user_input.get(feat, 0))
    
    diff_cols = [f'{feat}_diff' for feat in features if f'{feat}_diff' in df_copy.columns]
    df_copy['total_diff'] = df_copy[diff_cols].sum(axis=1)
    
    similar = df_copy.nsmallest(n, 'total_diff')[['number_of_bedrooms', 'number_of_bathrooms', 
                                                     'living_area', 'grade_of_the_house', 'price']]
    return similar


def create_price_distribution_chart(df, predicted_price=None):
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['price'],
        nbinsx=50,
        name='Properties',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    if predicted_price is not None:
        fig.add_vline(
            x=predicted_price,
            line_dash="dash",
            line_color="red",
            annotation_text="Your Prediction",
            annotation_position="top"
        )
    
    fig.update_layout(
        title="Price Distribution",
        xaxis_title="Price (₹)",
        yaxis_title="Count",
        showlegend=False,
        height=300
    )
    
    return fig



def create_feature_correlation_chart(df):
    """Create correlation chart for key features"""
    features = ['living_area', 'grade_of_the_house', 'number_of_bathrooms', 
                'number_of_bedrooms', 'price']
    
    corr_data = df[features].corr()['price'].drop('price').sort_values(ascending=True)
    
    fig = go.Figure(go.Bar(
        x=corr_data.values,
        y=corr_data.index,
        orientation='h',
        marker_color='steelblue'
    ))
    
    fig.update_layout(
        title="Feature Correlation with Price",
        xaxis_title="Correlation",
        yaxis_title="Feature",
        height=300
    )
    
    return fig


# --------------------------
# Main Application
# --------------------------

def main():
    # Header
    st.markdown('<p class="main-header">🏠 Housing Price Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Get accurate price estimates using machine learning</p>', unsafe_allow_html=True)
    
    # Load resources
    with st.spinner("Loading model and data..."):
        model, feature_cols, model_error = load_model_and_features()
        df, data_error = load_data()
    
    # Error handling
    if model_error:
        st.error(f"❌ Failed to load model: {model_error}")
        return
    
    if data_error:
        st.error(f"❌ Failed to load data: {data_error}")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.info(f"""
        **Model Type:** Random Forest Regressor  
        **Training Data:** {df.shape[0]:,} properties  
        **Features:** {len(feature_cols)} variables
        """)
        
        st.divider()
        
        st.header("📈 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Price", f"₹{df['price'].mean()/1e6:.2f}M")
        with col2:
            st.metric("Median Price", f"₹{df['price'].median()/1e6:.2f}M")
        
        st.metric("Price Range", f"₹{df['price'].min()/1e6:.1f}M - ₹{df['price'].max()/1e6:.1f}M")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Predict Price", "📊 Analytics", "📋 Dataset"])
    
    # ===== TAB 1: PREDICTION =====
    with tab1:
        st.subheader("Enter Property Details")
        
        # Quick presets
        with st.expander("💡 Use Quick Presets"):
            preset_col1, preset_col2, preset_col3 = st.columns(3)
            if preset_col1.button("🏠 Budget Home"):
                st.session_state.preset = "budget"
            if preset_col2.button("🏘️ Family Home"):
                st.session_state.preset = "family"
            if preset_col3.button("🏰 Luxury Home"):
                st.session_state.preset = "luxury"
        
        # Determine preset values
        preset = getattr(st.session_state, 'preset', None)
        if preset == "budget":
            default_bed, default_bath, default_area, default_grade = 2, 1.0, 1000, 5
        elif preset == "family":
            default_bed, default_bath, default_area, default_grade = 3, 2.0, 2000, 7
        elif preset == "luxury":
            default_bed, default_bath, default_area, default_grade = 5, 3.5, 4000, 10
        else:
            default_bed = int(df['number_of_bedrooms'].median())
            default_bath = float(df['number_of_bathrooms'].median())
            default_area = int(df['living_area'].median())
            default_grade = int(df['grade_of_the_house'].median())
        
        # Input Form - Organized by category
        st.markdown("### 🏠 Basic Features")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            number_of_bedrooms = st.number_input(
                "Bedrooms",
                min_value=int(df['number_of_bedrooms'].min()),
                max_value=int(df['number_of_bedrooms'].max()),
                value=default_bed,
                help="Number of bedrooms in the property"
            )
        
        with col2:
            number_of_bathrooms = st.number_input(
                "Bathrooms",
                min_value=float(df['number_of_bathrooms'].min()),
                max_value=float(df['number_of_bathrooms'].max()),
                value=default_bath,
                step=0.25,
                help="Number of bathrooms"
            )
        
        with col3:
            number_of_floors = st.number_input(
                "Floors",
                min_value=float(df['number_of_floors'].min()),
                max_value=float(df['number_of_floors'].max()),
                value=float(df['number_of_floors'].median()),
                step=0.5
            )
        
        with col4:
            waterfront_present = st.selectbox(
                "Waterfront",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                help="Does the property have a waterfront view?"
            )
        
        st.markdown("### 📏 Area & Space")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            living_area = st.number_input(
                "Living Area (sqft)",
                min_value=int(df['living_area'].min()),
                max_value=int(df['living_area'].max()),
                value=default_area,
                step=100
            )
        
        with col2:
            lot_area = st.number_input(
                "Lot Area (sqft)",
                min_value=int(df['lot_area'].min()),
                max_value=int(df['lot_area'].max()),
                value=int(df['lot_area'].median()),
                step=100
            )
        
        with col3:
            area_basement = st.number_input(
                "Basement Area (sqft)",
                min_value=0,
                max_value=int(df['area_of_the_basement'].max()),
                value=int(df['area_of_the_basement'].median()),
                step=50
            )
        
        st.markdown("### ⭐ Quality & Condition")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            grade_of_the_house = st.slider(
                "Grade (1-13)",
                min_value=int(df['grade_of_the_house'].min()),
                max_value=int(df['grade_of_the_house'].max()),
                value=default_grade,
                help="Overall grade given to the housing unit (1-13 scale)"
            )
        
        with col2:
            condition_of_the_house = st.slider(
                "Condition (1-5)",
                min_value=int(df['condition_of_the_house'].min()),
                max_value=int(df['condition_of_the_house'].max()),
                value=int(df['condition_of_the_house'].median()),
                help="Condition of the house (1=Poor, 5=Excellent)"
            )
        
        with col3:
            number_of_views = st.slider(
                "Views (0-4)",
                min_value=int(df['number_of_views'].min()),
                max_value=int(df['number_of_views'].max()),
                value=int(df['number_of_views'].median()),
                help="Number of times property has been viewed"
            )
        
        st.markdown("### 📅 Age & Renovation")
        col1, col2 = st.columns(2)
        
        with col1:
            built_year = st.slider(
                "Built Year",
                min_value=int(df['built_year'].min()),
                max_value=int(df['built_year'].max()),
                value=int(df['built_year'].median())
            )
        
        with col2:
            renovation_year = st.slider(
                "Renovation Year",
                min_value=int(df['renovation_year'].min()),
                max_value=int(df['renovation_year'].max()),
                value=int(df['renovation_year'].median()),
                help="Year of last renovation (0 if never renovated)"
            )
        
        st.markdown("### 📍 Location & Amenities")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            postal_code = st.selectbox(
                "Postal Code",
                options=sorted(df['postal_code'].unique())
            )
        
        with col2:
            number_of_schools_nearby = st.slider(
                "Nearby Schools",
                min_value=int(df['number_of_schools_nearby'].min()),
                max_value=int(df['number_of_schools_nearby'].max()),
                value=int(df['number_of_schools_nearby'].median())
            )
        
        with col3:
            distance_from_the_airport = st.slider(
                "Airport Distance (km)",
                min_value=float(df['distance_from_the_airport'].min()),
                max_value=float(df['distance_from_the_airport'].max()),
                value=float(df['distance_from_the_airport'].median())
            )
        
        # Additional fields (collapsed)
        with st.expander("🔧 Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                area_excl_basement = st.number_input(
                    "Area (excluding basement)",
                    min_value=0,
                    max_value=int(df['area_of_the_house(excluding_basement)'].max()),
                    value=int(df['area_of_the_house(excluding_basement)'].median())
                )
                lattitude = st.number_input(
                    "Latitude",
                    min_value=float(df['lattitude'].min()),
                    max_value=float(df['lattitude'].max()),
                    value=float(df['lattitude'].median()),
                    format="%.6f"
                )
                living_area_renov = st.number_input(
                    "Living Area (Renovated)",
                    min_value=0,
                    max_value=int(df['living_area_renov'].max()),
                    value=int(df['living_area_renov'].median())
                )
            with col2:
                longitude = st.number_input(
                    "Longitude",
                    min_value=float(df['longitude'].min()),
                    max_value=float(df['longitude'].max()),
                    value=float(df['longitude'].median()),
                    format="%.6f"
                )
                lot_area_renov = st.number_input(
                    "Lot Area (Renovated)",
                    min_value=0,
                    max_value=int(df['lot_area_renov'].max()),
                    value=int(df['lot_area_renov'].median())
                )
        
        # Build input dictionary
        raw_input = {
            "number_of_bedrooms": number_of_bedrooms,
            "number_of_bathrooms": number_of_bathrooms,
            "living_area": living_area,
            "lot_area": lot_area,
            "number_of_floors": number_of_floors,
            "waterfront_present": waterfront_present,
            "number_of_views": number_of_views,
            "condition_of_the_house": condition_of_the_house,
            "grade_of_the_house": grade_of_the_house,
            "area_of_the_house(excluding_basement)": area_excl_basement,
            "area_of_the_basement": area_basement,
            "built_year": built_year,
            "renovation_year": renovation_year,
            "postal_code": postal_code,
            "lattitude": lattitude,
            "longitude": longitude,
            "living_area_renov": living_area_renov,
            "lot_area_renov": lot_area_renov,
            "number_of_schools_nearby": number_of_schools_nearby,
            "distance_from_the_airport": distance_from_the_airport,
        }
        
        st.divider()
        
        # Prediction button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("🔮 Predict Price", type="primary", use_container_width=True)
        
        if predict_button:
            with st.spinner("Analyzing property features..."):
                X_input, preprocess_error = preprocess_input(raw_input, feature_cols)
                
                if preprocess_error:
                    st.error(f"❌ Error processing input: {preprocess_error}")
                else:
                    try:
                        pred_price = model.predict(X_input)[0]
                        
                        # Display prediction
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2 style="margin:0; font-size: 1.2rem; opacity: 0.9;">Estimated Price</h2>
                            <h1 style="margin:0.5rem 0; font-size: 3rem;">₹ {pred_price:,.0f}</h1>
                            <p style="margin:0; opacity: 0.9;">Based on {len(feature_cols)} property features</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Price category
                        q1, q3 = df['price'].quantile([0.33, 0.66])
                        if pred_price < q1:
                            category = "Budget-Friendly"
                            emoji = "💰"
                            color = "green"
                        elif pred_price < q3:
                            category = "Mid-Range"
                            emoji = "🏘️"
                            color = "blue"
                        else:
                            category = "Premium"
                            emoji = "✨"
                            color = "orange"
                        
                        st.markdown(f"### {emoji} Category: **:{color}[{category}]**")
                        
                        # Additional insights
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            percentile = (df['price'] < pred_price).mean() * 100
                            st.metric("Price Percentile", f"{percentile:.0f}%", 
                                     help="Your property is priced higher than this % of properties")
                        
                        with col2:
                            avg_price = df['price'].mean()
                            diff = ((pred_price - avg_price) / avg_price) * 100
                            st.metric("vs Average", f"{diff:+.1f}%",
                                     help="Difference from average market price")
                        
                        with col3:
                            price_per_sqft = pred_price / living_area if living_area > 0 else 0
                            st.metric("Price per sqft", f"₹{price_per_sqft:,.0f}")
                        
                        # Similar properties
                        st.markdown("### 🔍 Similar Properties in Database")
                        similar_props = get_similar_properties(df, raw_input, n=5)
                        st.dataframe(
                            similar_props.style.format({
                                'price': '₹{:,.0f}',
                                'living_area': '{:.0f} sqft'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")
    
    # ===== TAB 2: ANALYTICS =====
    with tab2:
        st.subheader("Market Analytics & Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = create_price_distribution_chart(df)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = create_feature_correlation_chart(df)
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("### 📊 Price by Key Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bedrooms vs Price
            bed_price = df.groupby('number_of_bedrooms')['price'].mean().reset_index()
            fig3 = px.bar(bed_price, x='number_of_bedrooms', y='price',
                         title='Average Price by Bedrooms',
                         labels={'price': 'Average Price (₹)', 'number_of_bedrooms': 'Bedrooms'})
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Grade vs Price
            grade_price = df.groupby('grade_of_the_house')['price'].mean().reset_index()
            fig4 = px.bar(grade_price, x='grade_of_the_house', y='price',
                         title='Average Price by Grade',
                         labels={'price': 'Average Price (₹)', 'grade_of_the_house': 'Grade'})
            st.plotly_chart(fig4, use_container_width=True)
    
    # ===== TAB 3: DATASET =====
    with tab3:
        st.subheader("📋 Training Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Properties", f"{len(df):,}")
        with col2:
            st.metric("Features", f"{len(df.columns)}")
        with col3:
            st.metric("Avg Price", f"₹{df['price'].mean()/1e6:.2f}M")
        with col4:
            st.metric("Max Price", f"₹{df['price'].max()/1e6:.2f}M")
        
        st.dataframe(df.head(100), use_container_width=True, height=400)
        
        with st.expander("📈 Dataset Statistics"):
            st.dataframe(df.describe(), use_container_width=True)


if __name__ == "__main__":
    main()