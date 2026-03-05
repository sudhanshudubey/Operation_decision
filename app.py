import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Set page config for a wider, professional layout
st.set_page_config(page_title="Retail Analytics Dashboard", layout="wide")

# -------------------------------------------------------------------------
# 1. LOAD AND PREP DATA (Cached so the website is fast!)
# -------------------------------------------------------------------------
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv('/Users/sudhanshudubey/Emlyon/SEM 2/Operation and Decision/group projet/dataset/dataset.csv')
    df['C520_DATE'] = pd.to_datetime(df['C520_DATE'])
    
    # Aggregate to daily level
    df_aggregated = df.groupby(['C520_DATE', 'ARTNRVIER', 'BEZ_LANG']).agg({
        'C520_SOLDPIECES': 'sum',      
        'C520_RETAILPRICE': 'mean',    
        'DISCOUNT': 'max'              
    }).reset_index()

    # Find top 10 products
    top_products = df_aggregated.groupby(['ARTNRVIER', 'BEZ_LANG'])['C520_SOLDPIECES'].sum().sort_values(ascending=False)
    top_10_names = top_products.head(10).reset_index()['BEZ_LANG'].tolist()
    
    # Build continuous time-series dictionary
    ts_data = {}
    df_model = df_aggregated[df_aggregated['BEZ_LANG'].isin(top_10_names)].copy()
    for prod in top_10_names:
        temp = df_model[df_model['BEZ_LANG'] == prod].sort_values('C520_DATE').set_index('C520_DATE')
        full_date_range = pd.date_range(temp.index.min(), temp.index.max(), freq='D')
        temp = temp.reindex(full_date_range)
        temp['C520_RETAILPRICE'] = temp['C520_RETAILPRICE'].ffill() 
        temp['DISCOUNT'] = temp['DISCOUNT'].fillna(0)               
        temp['C520_SOLDPIECES'] = temp['C520_SOLDPIECES'].fillna(0) 
        ts_data[prod] = temp
        
    return ts_data, top_10_names

ts_data, top_10_names = load_and_prep_data()

# -------------------------------------------------------------------------
# 2. DASHBOARD SIDEBAR & HEADER
# -------------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3081/3081648.png", width=100) # Optional generic retail icon
st.sidebar.title("Store Controls")
selected_product = st.sidebar.selectbox("Select a Product to Analyze:", top_10_names)

st.title("📊 Data-Driven Retail Analytics Dashboard")
st.markdown(f"**Current Product Focus:** `{selected_product}` | **Lead Time Assumption:** 2 Days")

# Create 3 Tabs for the 3 Assignment Steps
tab1, tab2, tab3 = st.tabs(["📈 Demand Forecasting", "💲 Price Elasticity", "📦 Inventory Policy"])

# -------------------------------------------------------------------------
# TAB 1: DEMAND FORECASTING
# -------------------------------------------------------------------------
with tab1:
    st.header(f"Forecasting Engine: {selected_product}")
    
    temp = ts_data[selected_product].copy()
    temp['lag_1'] = temp['C520_SOLDPIECES'].shift(1) 
    temp['lag_7'] = temp['C520_SOLDPIECES'].shift(7) 
    temp['rolling_7'] = temp['lag_1'].rolling(7).mean() 
    temp['dayofweek'] = temp.index.dayofweek 
    temp['month'] = temp.index.month         
    temp = temp.dropna()
    
    X = temp[['lag_1', 'lag_7', 'rolling_7', 'dayofweek', 'month', 'C520_RETAILPRICE', 'DISCOUNT']]
    y = temp['C520_SOLDPIECES'] 
    
    split_idx = int(len(temp) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train Models
    mae_naive = mean_absolute_error(y_test, X_test['lag_1'])
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, preds_rf)
    
    # Display Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Naive Baseline MAE", f"{mae_naive:.1f} units")
    col2.metric("Random Forest MAE", f"{mae_rf:.1f} units", delta=f"-{(mae_naive - mae_rf):.1f} error", delta_color="inverse")
    col3.metric("Model Improvement", f"{((mae_naive - mae_rf) / mae_naive * 100):.1f}%", "Outperforms Baseline")
    
    # Plot Actual vs Forecast
    st.subheader("Test Data Forecast vs. Actual Sales")
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(y_test.index, y_test.values, label='Actual Demand', color='gray', alpha=0.6)
    ax1.plot(y_test.index, preds_rf, label='RF Forecast', color='#4C72B0', linewidth=2)
    ax1.legend()
    ax1.set_ylabel("Units Sold")
    st.pyplot(fig1)

# -------------------------------------------------------------------------
# TAB 2: PRICE ELASTICITY
# -------------------------------------------------------------------------
with tab2:
    st.header(f"Price vs. Demand: {selected_product}")
    st.markdown("Are we losing sales when we raise prices? Let's look at the historical price quartiles.")
    
    prod_data = ts_data[selected_product].copy()
    try:
        prod_data['Price_Bin'] = pd.qcut(prod_data['C520_RETAILPRICE'], q=4, labels=['Q1 (Cheapest)', 'Q2', 'Q3', 'Q4 (Most Expensive)'], duplicates='drop')
        
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.boxplot(x='Price_Bin', y='C520_SOLDPIECES', data=prod_data, palette='Blues', ax=ax2)
        ax2.set_title(f"Demand Distribution across Price Tiers", fontsize=14)
        ax2.set_xlabel("Price Quartiles")
        ax2.set_ylabel("Daily Demand")
        st.pyplot(fig2)
        
        st.info("**Manager Insight:** Be careful interpreting this! Prices often rise in the winter due to supply shortages, which is also when organic demand for fresh produce naturally falls. Seasonality is a massive confounding variable here.")
    except ValueError:
        st.warning("Not enough price variation for this product to create distinct price bins.")

# -------------------------------------------------------------------------
# TAB 3: INVENTORY POLICY
# -------------------------------------------------------------------------
with tab3:
    st.header("The Cost of Reliability (Safety Stock Simulation)")
    st.markdown("Comparing the required safety stock to achieve 90% vs 95% service levels (Assumed Lead Time: 2 Days).")
    
    # Calculate for Top 5
    inv_results = []
    for prod in top_10_names[:5]:
        tmp = ts_data[prod].copy()
        tmp['demand_L2'] = tmp['C520_SOLDPIECES'].rolling(2).sum().dropna()
        mean_L2 = tmp['demand_L2'].mean()
        R_90 = tmp['demand_L2'].quantile(0.90)
        R_95 = tmp['demand_L2'].quantile(0.95)
        inv_results.append({'Product': prod, 'SS_90': R_90 - mean_L2, 'SS_95': R_95 - mean_L2})
        
    inv_df = pd.DataFrame(inv_results)
    
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(inv_df['Product']))
    width = 0.35
    ax3.bar(x - width/2, inv_df['SS_90'], width, label='90% Service Level', color='#4C72B0', edgecolor='black')
    ax3.bar(x + width/2, inv_df['SS_95'], width, label='95% Service Level', color='#C44E52', edgecolor='black')
    ax3.set_ylabel('Safety Stock (Units)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(inv_df['Product'])
    ax3.legend()
    st.pyplot(fig3)
    
    st.error("**Manager Insight:** Notice the red bars. For high-variance items, bumping our service level from 90% to 95% nearly triples our safety stock. For fresh produce, we highly recommend capping the target at 90% to prevent massive food spoilage.")