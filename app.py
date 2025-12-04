
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import sqlite3
import warnings
import requests
import json
import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================

st.set_page_config(layout="wide", page_title="MA Grant Fairness Audit", page_icon="⚖️")

# Database Paths
DB_USASPENDING = 'usaspending_lite.db'
DB_CENSUS = 'census_data_ma.db'

# MassGIS API
URL_MASSGIS_TOWNS = "https://arcgisserver.digital.mass.gov/arcgisserver/rest/services/AGOL/Towns_survey_polym/FeatureServer/0/query"

# --- MAPS ---
CITY_NAME_MANUAL_MAP = {
    "ALLSTON": "BOSTON", "BRIGHTON": "BOSTON", "DORCHESTER": "BOSTON", "JAMAICA PLAIN": "BOSTON",
    "ROXBURY CROSSING": "BOSTON", "DORCHESTER CENTER": "BOSTON", "WEST ROXBURY": "BOSTON",
    "ROXBURY": "BOSTON", "MATTAPAN": "BOSTON", "ROSLINDALE": "BOSTON", "SOUTH BOSTON": "BOSTON",
    "EAST BOSTON": "BOSTON", "CHARLESTOWN": "BOSTON", "NEWTON HIGHLANDS": "NEWTON",
    "NEWTONVILLE": "NEWTON", "NEWTON CENTER": "NEWTON", "NEWTON LOWER FALLS": "NEWTON",
    "AUBURNDALE": "NEWTON", "WABAN": "NEWTON", "CHESTNUT HILL": "NEWTON",
    "NEWTON UPPER FALLS": "NEWTON", "EAST TAUNTON": "TAUNTON", "NEEDHAM HEIGHTS": "NEEDHAM",
    "SOUTH LANCASTER": "LANCASTER", "SHELBURNE FALLS": "BUCKLAND", "TURNERS FALLS": "MONTAGUE",
    "SOUTH DEERFIELD": "DEERFIELD", "MIDDLEBORO": "MIDDLEBOROUGH", "POCASSET": "BOURNE",
    "BUZZARDS BAY": "BOURNE", "HYANNIS": "BARNSTABLE", "CENTERVILLE": "BARNSTABLE",
    "WEST BARNSTABLE": "BARNSTABLE", "EAST FALMOUTH": "FALMOUTH", "WOODS HOLE": "FALMOUTH",
    "NORTH FALMOUTH": "FALMOUTH", "WELLESLEY HILLS": "WELLESLEY", "GILBERTVILLE": "HARDWICK",
    "HARWICH PORT": "HARWICH", "WEST DENNIS": "YARMOUTH", "SOUTH YARMOUTH": "YARMOUTH",
    "YARMOUTH PORT": "YARMOUTH", "HANSCOM AFB": "BEDFORD", "INDIAN ORCHARD": "SPRINGFIELD",
    "FOXBORO": "FOXBOROUGH", "NORTH ATTLEBORO": "NORTH ATTLEBOROUGH",
    "MANCHESTER": "MANCHESTER-BY-THE-SEA", "EAST WEYMOUTH": "WEYMOUTH",
    "WHITINSVILLE": "NORTHBRIDGE", "NORTH BILLERICA": "BILLERICA", "VINEYARD HAVEN": "TISBURY",
    "DEVENS": "AYER", "NORTH ANDOVER": "ANDOVER", "EASTHAM": "NORTH EASTHAM",
    "WEST TISBURY": "TISBURY", "WAREHAM": "WEST WAREHAM", "WEST HARWICH": "HARWICH",
    "EAST BRIDGEWATER": "BRIDGEWATER", "SOUTH LEE": "LEE", "WEST BRIDGEWATER": "BRIDGEWATER",
    "WEST NEWTON": "NEWTON", "EAST LONGMEADOW": "LONGMEADOW", "NORTH READING": "READING",
    "SOUTH CHATHAM": "CHATHAM", "NORTH OXFORD": "OXFORD", "EAST WAREHAM": "WEST WAREHAM",
    "NORTH DARTMOUTH": "DARTMOUTH", "SOUTH DARTMOUTH": "DARTMOUTH", "HYDE PARK": "BOSTON",
    "READVILLE": "BOSTON", "FLORENCE": "NORTHAMPTON", "NORTH EASTON": "EASTON",
    "SOUTH EASTON": "EASTON", "NORTH GRAFTON": "GRAFTON", "SOUTH GRAFTON": "GRAFTON",
    "TYNGSBORO": "TYNGSBOROUGH", "SOUTH HAMILTON": "HAMILTON", "OSTERVILLE": "BARNSTABLE",
    "COTUIT": "MASHPEE", "SAGAMORE BEACH": "BOURNE", "ONSET": "WAREHAM",
    "THREE RIVERS": "PALMER", "BONDSVILLE": "PALMER", "FEEDING HILLS": "AGAWAM",
    "MILL RIVER": "WESTFIELD", "BYFIELD": "NEWBURY", "NORTH CHELMSFORD": "CHELMSFORD",
    "NORTH TRURO": "TRURO", "SIASCONSET": "NANTUCKET", "CUTTYHUNK": "GOSNOLD",
    "HAYDENVILLE": "WILLIAMSBURG", "HATHORNE": "DANVERS", "ASHLEY FALLS": "SHEFFIELD"
}

CITY_TO_COUNTY_STRICT = {
    "WATERTOWN": "MIDDLESEX", "BRAINTREE": "NORFOLK", "EASTHAMPTON": "HAMPSHIRE",
    "BRIDGEWATER": "PLYMOUTH", "SOUTHBRIDGE": "WORCESTER", "FRANKLIN": "NORFOLK",
    "AMHERST": "HAMPSHIRE", "NORTHAMPTON": "HAMPSHIRE", "AGAWAM": "HAMPDEN",
    "METHUEN": "ESSEX", "WEYMOUTH": "NORFOLK", "PEABODY": "ESSEX",
    "ATTLEBORO": "BRISTOL", "WESTFIELD": "HAMPDEN", "WOBURN": "MIDDLESEX",
    "HOLYOKE": "HAMPDEN", "CHELSEA": "SUFFOLK", "REVERE": "SUFFOLK",
    "WINTHROP": "SUFFOLK", "MELROSE": "MIDDLESEX", "WOODS HOLE": "BARNSTABLE",
    "CHESTNUT HILL": "MIDDLESEX", "NORTH DARTMOUTH": "BRISTOL", "NEWTON CENTER": "MIDDLESEX",
    "ROXBURY CROSSING": "SUFFOLK", "WELLESLEY HILLS": "NORFOLK", "AUBURNDALE": "MIDDLESEX",
    "NEEDHAM HEIGHTS": "NORFOLK", "BUZZARDS BAY": "BARNSTABLE", "NEWTON HIGHLANDS": "MIDDLESEX",
    "MIDDLEBORO": "PLYMOUTH", "SOUTH DEERFIELD": "FRANKLIN", "NEWTONVILLE": "MIDDLESEX",
    "EAST FALMOUTH": "BARNSTABLE", "TURNERS FALLS": "FRANKLIN", "VINEYARD HAVEN": "DUKES",
    "DORCHESTER CENTER": "SUFFOLK", "SOUTH YARMOUTH": "BARNSTABLE", "EAST WEYMOUTH": "NORFOLK",
    "POCASSET": "BARNSTABLE", "HANSCOM AFB": "MIDDLESEX", "DEVENS": "WORCESTER",
    "HYANNIS": "BARNSTABLE", "DORCHESTER": "SUFFOLK", "ROXBURY": "SUFFOLK",
    "MATTAPAN": "SUFFOLK", "JAMAICA PLAIN": "SUFFOLK", "CHARLESTOWN": "SUFFOLK",
    "BRIGHTON": "SUFFOLK", "ALLSTON": "SUFFOLK", "SOUTH BOSTON": "SUFFOLK",
    "EAST BOSTON": "SUFFOLK", "WEST ROXBURY": "SUFFOLK", "ROSLINDALE": "SUFFOLK",
    "HYDE PARK": "SUFFOLK", "CAMBRIDGE": "MIDDLESEX", "SOMERVILLE": "MIDDLESEX",
    "BROOKLINE": "NORFOLK", "QUINCY": "NORFOLK", "LYNN": "ESSEX",
    "LOWELL": "MIDDLESEX", "LAWRENCE": "ESSEX", "FALL RIVER": "BRISTOL",
    "NEW BEDFORD": "BRISTOL", "BROCKTON": "PLYMOUTH", "SPRINGFIELD": "HAMPDEN",
    "BOSTON": "SUFFOLK", "WORCESTER": "WORCESTER", "BARNSTABLE": "BARNSTABLE",
    "PITTSFIELD": "BERKSHIRE", "SALEM": "ESSEX", "HAVERHILL": "ESSEX",
    "TAUNTON": "BRISTOL", "PLYMOUTH": "PLYMOUTH", "FALMOUTH": "BARNSTABLE",
    "MEDFORD": "MIDDLESEX", "MALDEN": "MIDDLESEX", "WALTHAM": "MIDDLESEX",
    "FRAMINGHAM": "MIDDLESEX", "CHICOPEE": "HAMPDEN", "LEOMINSTER": "WORCESTER",
    "FITCHBURG": "WORCESTER", "BEVERLY": "ESSEX", "MARLBOROUGH": "MIDDLESEX",
    "EVERETT": "MIDDLESEX", "ARLINGTON": "MIDDLESEX", "BILLERICA": "MIDDLESEX",
    "ANDOVER": "ESSEX", "NORTH ANDOVER": "ESSEX", "GLOUCESTER": "ESSEX",
    "DANVERS": "ESSEX", "SAUGUS": "ESSEX", "TEWKSBURY": "MIDDLESEX",
    "DARTMOUTH": "BRISTOL", "YARMOUTH": "BARNSTABLE", "NANTUCKET": "NANTUCKET",
    "EDGARTOWN": "DUKES", "OAK BLUFFS": "DUKES", "TISBURY": "DUKES",
    "GREENFIELD": "FRANKLIN"
}

# ==============================================================================
# 2. GEOSPATIAL ENGINE
# ==============================================================================

MA_COUNTIES_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "properties": {"name": "SUFFOLK"}, "geometry": {"type": "Polygon", "coordinates": [[[-71.191, 42.34], [-71.02, 42.45], [-70.92, 42.38], [-71.05, 42.22], [-71.191, 42.34]]]}},
        {"type": "Feature", "properties": {"name": "MIDDLESEX"}, "geometry": {"type": "Polygon", "coordinates": [[[-71.8, 42.7], [-71.3, 42.75], [-71.05, 42.45], [-71.35, 42.2], [-71.8, 42.7]]]}},
        {"type": "Feature", "properties": {"name": "WORCESTER"}, "geometry": {"type": "Polygon", "coordinates": [[[-72.2, 42.7], [-71.8, 42.7], [-71.5, 42.1], [-72.1, 42.05], [-72.2, 42.7]]]}},
        {"type": "Feature", "properties": {"name": "ESSEX"}, "geometry": {"type": "Polygon", "coordinates": [[[-71.2, 42.8], [-70.8, 42.88], [-70.6, 42.6], [-71.0, 42.45], [-71.2, 42.8]]]}},
        {"type": "Feature", "properties": {"name": "NORFOLK"}, "geometry": {"type": "Polygon", "coordinates": [[[-71.4, 42.3], [-71.0, 42.3], [-70.9, 42.15], [-71.4, 42.05], [-71.4, 42.3]]]}},
        {"type": "Feature", "properties": {"name": "BRISTOL"}, "geometry": {"type": "Polygon", "coordinates": [[[-71.4, 42.05], [-71.0, 42.05], [-70.9, 41.5], [-71.15, 41.45], [-71.4, 42.05]]]}},
        {"type": "Feature", "properties": {"name": "PLYMOUTH"}, "geometry": {"type": "Polygon", "coordinates": [[[-71.0, 42.2], [-70.6, 42.25], [-70.5, 41.75], [-70.9, 41.7], [-71.0, 42.2]]]}},
        {"type": "Feature", "properties": {"name": "BARNSTABLE"}, "geometry": {"type": "Polygon", "coordinates": [[[-70.7, 41.8], [-70.0, 42.1], [-69.9, 41.6], [-70.6, 41.5], [-70.7, 41.8]]]}},
        {"type": "Feature", "properties": {"name": "HAMPDEN"}, "geometry": {"type": "Polygon", "coordinates": [[[-73.0, 42.25], [-72.2, 42.25], [-72.2, 42.02], [-73.0, 42.02], [-73.0, 42.25]]]}},
        {"type": "Feature", "properties": {"name": "HAMPSHIRE"}, "geometry": {"type": "Polygon", "coordinates": [[[-73.0, 42.45], [-72.2, 42.45], [-72.2, 42.25], [-73.0, 42.25], [-73.0, 42.45]]]}},
        {"type": "Feature", "properties": {"name": "BERKSHIRE"}, "geometry": {"type": "Polygon", "coordinates": [[[-73.5, 42.75], [-73.0, 42.75], [-73.0, 42.02], [-73.5, 42.02], [-73.5, 42.75]]]}},
        {"type": "Feature", "properties": {"name": "FRANKLIN"}, "geometry": {"type": "Polygon", "coordinates": [[[-73.0, 42.75], [-72.2, 42.75], [-72.2, 42.45], [-73.0, 42.45], [-73.0, 42.75]]]}},
        {"type": "Feature", "properties": {"name": "DUKES"}, "geometry": {"type": "Polygon", "coordinates": [[[-70.8, 41.5], [-70.4, 41.5], [-70.4, 41.3], [-70.8, 41.3], [-70.8, 41.5]]]}},
        {"type": "Feature", "properties": {"name": "Nantucket"}, "geometry": {"type": "Polygon", "coordinates": [[[-70.3, 41.4], [-69.9, 41.4], [-69.9, 41.2], [-70.3, 41.2], [-70.3, 41.4]]]}}
    ]
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_massgis_towns():
    params = {'where': '1=1', 'outFields': 'TOWN,TOWN_ID', 'outSR': '4326', 'f': 'geojson', 'geometryPrecision': '5'}
    try:
        r = requests.get(URL_MASSGIS_TOWNS, params=params, timeout=20)
        if r.status_code == 200: return r.json()
        return {"type": "FeatureCollection", "features": []} 
    except:
        return {"type": "FeatureCollection", "features": []}

# ==============================================================================
# 3. DATA PIPELINE & LOGIC
# ==============================================================================

@st.cache_data
def load_and_process_data(start_date, end_date):
    # ... [Standard Logic] ...
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    conn = sqlite3.connect(DB_USASPENDING)
    try:
        query_trans = f"SELECT * FROM temp_aggregated_transactions WHERE \"Action Date\" BETWEEN '{start_str}' AND '{end_str}'"
        df_trans = pd.read_sql_query(query_trans, conn)
        query_lean = "SELECT * FROM project_lean_data"
        df_lean = pd.read_sql_query(query_lean, conn)
    except Exception as e:
        st.error(f"DB Error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

    if df_trans.empty: return pd.DataFrame()

    df_trans['Transaction Amount'] = pd.to_numeric(df_trans['Transaction Amount'], errors='coerce').fillna(0)
    df_trans = df_trans.sort_values('Action Date')
    
    pos = df_trans[df_trans['Transaction Amount'] > 0].groupby('Award ID')['Transaction Amount'].sum().reset_index(name='Total_Positive_Amount')
    neg = df_trans[df_trans['Transaction Amount'] < 0].groupby('Award ID')['Transaction Amount'].sum().reset_index(name='Total_Negative_Amount')
    last = df_trans.groupby('Award ID').last().reset_index()
    last = last.rename(columns={'Action Date': 'Last_Action_Date', 'Action Type': 'Last_Action_Type', 'Transaction Amount': 'Last_Transaction_Amount'})
    
    df_agg = last.merge(pos, on='Award ID', how='left').merge(neg, on='Award ID', how='left')
    df_agg['Total_Positive_Amount'] = df_agg.get('Total_Positive_Amount', pd.Series(0)).fillna(0)
    df_agg['Total_Negative_Amount'] = df_agg.get('Total_Negative_Amount', pd.Series(0)).fillna(0)
    combined = df_agg.merge(df_lean, left_on='Award ID', right_on='award_id_fain', how='inner')
    
    combined['total_obligated_amount'] = pd.to_numeric(combined['total_obligated_amount'], errors='coerce').fillna(0)
    combined['total_outlayed_amount'] = pd.to_numeric(combined['total_outlayed_amount'], errors='coerce').fillna(0)
    
    cond_rescinded = (combined['Total_Negative_Amount'] < 0)
    cond_cancelled_1 = (combined['total_outlayed_amount'] == 0) & (combined['total_obligated_amount'] == 0)
    cond_cancelled_2 = (combined['total_outlayed_amount'] > 0) & (combined['Last_Transaction_Amount'] == 0) & (combined['Last_Action_Type'].isin(['C','D','E']))
    cond_cancelled = (cond_cancelled_1 | cond_cancelled_2) & (~cond_rescinded)
    combined['Category'] = np.select([cond_rescinded, cond_cancelled], ['Rescinded', 'Cancelled'], default='Healthy')
    
    def calc_impact(row):
        if row['Category'] == 'Rescinded': return abs(row['Total_Negative_Amount'])
        if row['Category'] == 'Cancelled': return max(0, row['total_obligated_amount'] - row['total_outlayed_amount'])
        return 0
    combined['impact_amount'] = combined.apply(calc_impact, axis=1)
    combined['city_clean_raw'] = combined['recipient_city_name'].astype(str).str.upper().str.strip()
    combined['city_clean'] = combined['city_clean_raw'].replace(CITY_NAME_MANUAL_MAP)
    
    city_agg = combined.groupby('city_clean').agg(
        financial_impact=('impact_amount', 'sum'),
        total_obligated=('total_obligated_amount', 'sum'),
        total_grants=('Award ID', 'count')
    ).reset_index()
    
    city_agg['termination_rate'] = np.where(city_agg['total_obligated'] > 0, city_agg['financial_impact'] / city_agg['total_obligated'], 0.0).clip(0, 1)

    conn_census = sqlite3.connect(DB_CENSUS)
    try:
        df_census = pd.read_sql("SELECT * FROM census_combined_ma_places", conn_census)
        df_census['city_clean'] = df_census['city_clean'].astype(str).str.upper().str.strip()
        pop_col = 'population_2020' if 'population_2020' in df_census.columns else 'total_population'
        if pop_col not in df_census.columns: df_census[pop_col] = 1
        df_census['population_2020'] = df_census[pop_col]
        if 'people_of_color_pct' not in df_census.columns: df_census['people_of_color_pct'] = 0.0
        df_census['poc_rate'] = df_census['people_of_color_pct'] / 100.0
        df_census['poverty_rate'] = df_census['poverty_rate'] / 100.0 if 'poverty_rate' in df_census.columns else 0.0
        df_census = df_census.sort_values('population_2020', ascending=False).drop_duplicates(subset=['city_clean'])
    finally:
        conn_census.close()
        
    df_final = pd.merge(city_agg, df_census, on='city_clean', how='left')
    df_final = df_final.rename(columns={'city_clean': 'municipality'})
    df_final['county'] = df_final['municipality'].map(CITY_TO_COUNTY_STRICT).fillna('UNKNOWN')
    
    # Fill NaNs
    for col in ['poc_rate', 'poverty_rate', 'termination_rate', 'financial_impact']:
        df_final[col] = df_final[col].fillna(0)
    
    return df_final

def aggregate_data(df, granularity):
    if df.empty: return pd.DataFrame()
    if granularity == 'Municipality Level': return df
    elif granularity == 'County Level':
        df_county = df[df['county'] != 'UNKNOWN'].copy()
        grouped = df_county.groupby('county').apply(
            lambda x: pd.Series({
                'financial_impact': x['financial_impact'].sum(),
                'total_grants': x['total_grants'].sum(),
                'poc_rate': np.average(x['poc_rate'], weights=x['population_2020']) if x['population_2020'].sum() > 0 else 0,
                'poverty_rate': np.average(x['poverty_rate'], weights=x['population_2020']) if x['population_2020'].sum() > 0 else 0,
                'termination_rate': np.average(x['termination_rate'], weights=x['total_obligated']) if x['total_obligated'].sum() > 0 else 0,
            })
        ).reset_index()
        grouped['municipality'] = grouped['county']
        
        grouped['poc_rate'] = grouped['poc_rate'].fillna(0)
        grouped['poverty_rate'] = grouped['poverty_rate'].fillna(0)
        
        return grouped
    return df

# ==============================================================================
# 4. FRONTEND
# ==============================================================================

def main():
    st.title("⚖️ MA Grant Fairness Audit Dashboard")
    st.markdown("**Administrative Fairness Audit** | Real Boundaries & Impact Analysis (2D View)")

    st.sidebar.header("⚙️ Settings")
    
    st.sidebar.subheader("📅 Timeframe Analysis")
    min_date = datetime.date(2020, 1, 1)
    max_date = datetime.date(2025, 12, 31)
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    
    if start_date > end_date:
        st.sidebar.error("Error: End Date must be after Start Date.")
        st.stop()

    with st.spinner(f"Loading Data ({start_date} to {end_date})..."):
        df_raw = load_and_process_data(start_date, end_date)

    if df_raw.empty:
        st.warning("No impact data found for this period.")
        st.stop()

    granularity = st.sidebar.radio("View Mode", ["County Level", "Municipality Level"])
    
    # --- NEW MAP MODE: 3D INTERSECTIONALITY ---
    map_style = st.sidebar.radio(
        "Map Style", 
        ["Real Boundaries (Choropleth)", "Bubble Map (Dual Layer)", "3D Impact x Demographics"]
    )
    
    exclude_boston = st.sidebar.checkbox("Exclude Boston (Outlier)", value=False)
    
    # Dynamic Metric Selection
    metric_key_map = {
        "Financial Impact ($)": "financial_impact", 
        "Termination Rate (%)": "termination_rate", 
        "POC Rate": "poc_rate", 
        "Poverty Rate": "poverty_rate"
    }

    if "Real Boundaries" in map_style:
        # Single Metric for Choropleth
        color_metric = st.sidebar.selectbox(
            "Map Color Metric", 
            list(metric_key_map.keys()), 
            index=0
        )
        impact_metric_name = color_metric # For consistency in variable names
        demo_metric_name = None
    else:
        # Dual Metric for Bubble/3D
        st.sidebar.subheader("📊 Dual-Layer Settings")
        demo_metric_name = st.sidebar.selectbox(
            "Base Layer (Color)", 
            ["POC Rate", "Poverty Rate"], 
            index=0,
            help="Determines the color intensity (Blue to Red)."
        )
        impact_metric_name = st.sidebar.selectbox(
            "Impact Layer (Size/Height)", 
            ["Financial Impact ($)", "Termination Rate (%)"], 
            index=0,
            help="Determines the radius of bubbles or height of 3D bars."
        )
    
    df_agg = aggregate_data(df_raw, granularity)
    
    if exclude_boston:
        df_agg = df_agg[df_agg['municipality'] != 'BOSTON']
        if granularity == 'County Level': df_agg = df_agg[df_agg['municipality'] != 'SUFFOLK']

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Financial Impact", f"${df_agg['financial_impact'].sum()/1e6:.1f}M", delta="Visible Loss")
    m2.metric("Avg Termination Rate", f"{df_agg['termination_rate'].mean()*100:.1f}%")
    m3.metric("Most Impacted", df_agg.loc[df_agg['financial_impact'].idxmax()]['municipality'] if not df_agg.empty else "N/A")

    if granularity == 'County Level':
        geo_data = MA_COUNTIES_GEOJSON
        key_prop = 'name'
    else:
        with st.spinner("Fetching High-Res MassGIS Town Boundaries..."):
            geo_data = fetch_massgis_towns()
        key_prop = 'TOWN'

    data_lookup = df_agg.set_index('municipality').to_dict('index')
    
    # Resolve Column Names
    if "Real Boundaries" in map_style:
        target_col = metric_key_map[color_metric]
        demo_col = None
    else:
        target_col = metric_key_map[impact_metric_name]
        demo_col = metric_key_map[demo_metric_name]
    
    def get_centroid(geometry):
        if geometry['type'] == 'Polygon':
            coords = geometry['coordinates'][0]
            return [sum(x[0] for x in coords)/len(coords), sum(x[1] for x in coords)/len(coords)]
        elif geometry['type'] == 'MultiPolygon':
            coords = geometry['coordinates'][0][0]
            return [sum(x[0] for x in coords)/len(coords), sum(x[1] for x in coords)/len(coords)]
        return [-71.0, 42.0]

    # --- PREPARE DATA FOR RENDERING ---
    # Pre-calculate max values for normalization
    # Impact Normalization (Log for $, Linear for %)
    if 'Financial' in impact_metric_name:
        impact_values = df_agg[target_col].fillna(0).values
        log_impact = np.log1p(impact_values)
        max_val_impact = log_impact.max() if log_impact.max() > 0 else 1
        is_log = True
    else:
        # Rate is already 0-1, but let's normalize to max in dataset for visibility or keep absolute?
        # Keeping absolute 0-1 is better for rates, but for size we might want relative scaling.
        # Let's use relative to max to make dots visible even if rates are low overall.
        max_val_impact = df_agg[target_col].max() if df_agg[target_col].max() > 0 else 1
        is_log = False

    features_to_render = []
    
    for feature in geo_data['features']:
        g_name = feature['properties'].get(key_prop, '').upper().strip()
        
        if g_name in data_lookup:
            props = data_lookup[g_name]
            feature['properties'].update(props)
            feature['properties']['name'] = g_name
            
            # --- Calculate Render Properties ---
            
            # Pre-format strings for tooltip (PyDeck doesn't support f-string formatting in JS)
            raw_fin = props.get('financial_impact', 0)
            raw_term = props.get('termination_rate', 0)
            raw_poc = props.get('poc_rate', 0)
            raw_pov = props.get('poverty_rate', 0)
            
            feature['properties']['fmt_financial'] = f"${raw_fin:,.0f}"
            feature['properties']['fmt_termination'] = f"{raw_term:.1%}"
            feature['properties']['fmt_poc'] = f"{raw_poc:.1%}"
            feature['properties']['fmt_poverty'] = f"{raw_pov:.1%}"
            
            # 1. IMPACT (Size / Elevation)
            val_impact = props.get(target_col, 0)
            if is_log:
                norm_impact = np.log1p(val_impact) / max_val_impact
            else:
                norm_impact = val_impact / max_val_impact
            
            if np.isnan(norm_impact) or norm_impact < 0: norm_impact = 0
            
            # 2. DEMOGRAPHICS (Color) - Only for Dual Modes
            if demo_col:
                val_demo = props.get(demo_col, 0)
                # Demographics are 0.0 - 1.0. 
                # We map 0.0 -> Blue, 1.0 -> Red.
                # We can just use the raw rate as 't'.
                t = val_demo 
                r = int(255 * t)
                b = int(255 * (1 - t))
                feature['properties']['dual_color'] = [r, 0, b, 200]
            
            # --- Assign to Feature ---
            
            # Elevation (3D)
            feature['properties']['elevation'] = norm_impact * 50000 
            
            # Radius (Bubble) - Scale based on impact
            feature['properties']['radius'] = 300 + (norm_impact * 12000)
            
            # Centroid
            feature['properties']['centroid'] = get_centroid(feature['geometry'])
            
            # Color Logic
            if "Real Boundaries" in map_style:
                # Single Metric Logic
                val = props.get(target_col, 0)
                # Normalize generic
                if 'Financial' in color_metric:
                     norm_single = np.log1p(val) / (np.log1p(df_agg[target_col].max()) if df_agg[target_col].max() > 0 else 1)
                else:
                     norm_single = val / (df_agg[target_col].max() if df_agg[target_col].max() > 0 else 1)
                
                if np.isnan(norm_single): norm_single = 0
                
                # Red scale for Impact/Rate, Blue for Demo (inverted?) - Let's keep simple Heatmap style
                # Actually, user wanted consistent: Impact = Red, Demo = Blue.
                if 'Impact' in color_metric or 'Rate' in color_metric: # Termination Rate
                    c = int(255 * (1 - norm_single))
                    feature['properties']['fill_rgb'] = [255, c, c, 200] # Red Gradient
                else: # POC / Poverty
                    # High POC = Dark Blue? Or High POC = Red (Risk)?
                    # Usually High POC = Focus. Let's match 3D: High = Red? 
                    # Previous code: High POC = Blue [50,50,255]?? No, let's look at previous code.
                    # Previous: High val -> c is small. [c, c, 255]. So High val = [0,0,255] (Blue).
                    # Let's switch to Heatmap style (Red) for all "Heat" maps? 
                    # Or keep Blue for demographics to distinguish.
                    c = int(255 * (1 - norm_single))
                    feature['properties']['fill_rgb'] = [c, c, 255, 200] # Blue Gradient
            else:
                # Dual Mode Colors
                feature['properties']['fill_rgb'] = feature['properties']['dual_color']
                feature['properties']['dot_rgb'] = feature['properties']['dual_color']
                # For 3D, fill_color is used. For Bubble, dot_rgb is used.

            features_to_render.append(feature)
        else:
            # Missing Data
            feature['properties']['fill_rgb'] = [240, 240, 240, 100]
            feature['properties']['dual_color'] = [200, 200, 200, 100]
            feature['properties']['dot_rgb'] = [200, 200, 200, 100]
            feature['properties']['elevation'] = 100
            feature['properties']['radius'] = 100
            feature['properties']['centroid'] = get_centroid(feature['geometry'])
            feature['properties']['name'] = g_name
            
            feature['properties']['fmt_financial'] = "$0"
            feature['properties']['fmt_termination'] = "0.0%"
            feature['properties']['fmt_poc'] = "0.0%"
            feature['properties']['fmt_poverty'] = "0.0%"
            
            features_to_render.append(feature)

    geo_data['features'] = features_to_render

    # --- RENDER ---
    if "3D" in map_style:
        view_state = pdk.ViewState(latitude=42.1, longitude=-71.5, zoom=8, pitch=60, bearing=30)
        layer = pdk.Layer(
            "GeoJsonLayer",
            geo_data,
            opacity=0.9,
            stroked=False,
            filled=True,
            extruded=True,
            wireframe=True,
            get_elevation="properties.elevation",
            get_fill_color="properties.dual_color",
            pickable=True
        )
        tooltip_text = f"<b>{{name}}</b><br>Height ({impact_metric_name}): {{fmt_financial}} / {{fmt_termination}}<br>Color ({demo_metric_name}): {{fmt_poc}} / {{fmt_poverty}}"
        
    elif "Bubble" in map_style:
        view_state = pdk.ViewState(latitude=42.1, longitude=-71.5, zoom=8, pitch=0)
        layer = pdk.Layer(
            "ScatterplotLayer",
            geo_data['features'],
            get_position="properties.centroid",
            get_radius="properties.radius",
            get_fill_color="properties.dot_rgb",
            pickable=True, opacity=0.8, stroked=True, filled=True,
            radius_scale=1, radius_min_pixels=3, radius_max_pixels=100,
            get_line_color=[50, 50, 50], line_width_min_pixels=1
        )
        tooltip_text = f"<b>{{name}}</b><br>Size ({impact_metric_name}): {{fmt_financial}} / {{fmt_termination}}<br>Color ({demo_metric_name}): {{fmt_poc}} / {{fmt_poverty}}"
    else:
        view_state = pdk.ViewState(latitude=42.1, longitude=-71.5, zoom=8, pitch=0)
        layer = pdk.Layer(
            "GeoJsonLayer",
            geo_data,
            opacity=0.8, stroked=True, filled=True, extruded=False, wireframe=True,
            get_fill_color="properties.fill_rgb", get_line_color=[100, 100, 100], line_width_min_pixels=1, pickable=True
        )
        
        # Helper to select correct pre-formatted field based on user selection
        fmt_lookup = {
            "Financial Impact ($)": "{fmt_financial}",
            "Termination Rate (%)": "{fmt_termination}",
            "POC Rate": "{fmt_poc}",
            "Poverty Rate": "{fmt_poverty}"
        }
        fmt_val = fmt_lookup.get(color_metric, "{"+target_col+"}")
        
        tooltip_text = f"<b>{{name}}</b><br>{color_metric}: {fmt_val}"

    st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=view_state, layers=[layer], tooltip={"html": tooltip_text}))
    st.dataframe(df_agg[['municipality', 'financial_impact', 'termination_rate', 'poc_rate', 'poverty_rate']].sort_values('financial_impact', ascending=False), use_container_width=True)

if __name__ == "__main__":
    main()
