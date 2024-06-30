import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from supabase import create_client, Client
from statsforecast import StatsForecast
from statsforecast.models import CrostonOptimized


# Initialize connection to db
@st.cache_resource
def init_connection():
    url: str = st.secrets["supabase_url"]
    key: str = st.secrets["supabase_key"]
    client: Client = create_client(url, key)
    return client


# Run the function to make the connection
supabase = init_connection()


# Function to query the db
# Return all data
@st.cache_data(ttl=600)  # cache clears after 10 minutes
def run_query():
    return supabase.table("car_parts_monthly_sales").select("*").execute().data


# Function to create a Dataframe
# Make sure that volume is an integer
# Return dataframe
@st.cache_data(ttl=600)
def create_dataframe():
    rows = run_query()
    df = pd.json_normalize(rows)
    df["volume"] = df["volume"].astype(int)

    return df


# Function to plot data
@st.cache_data
def plot_volume(ids):
    fig, ax = plt.subplots()

    for id in ids:
        x = df[df["parts_id"] == id]['date']
        ax.plot(x,
                df[df['parts_id'] == id]['volume'], label=id)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.legend(loc='best')
    fig.autofmt_xdate()

    st.pyplot(fig)


# Function to format the dataframe as expected
# by statsforecast
@st.cache_data
def format_dataset(ids):
    model_df = df[df['parts_id'].isin(ids)]
    model_df = model_df.drop(['id'], axis=1)
    model_df.rename({"parts_id": "unique_id", "date": "ds",
                    "volume": "y"}, axis=1, inplace=True)

    return model_df


# Create the statsforecast object to train the model
# Return the statsforecast object
@st.cache_resource
def create_sf_object(model_df):
    models = [CrostonOptimized()]
    sf = StatsForecast(
        df=model_df,
        models=models,
        freq="MS",
        n_jobs=-1
    )
    return sf


# Function to make predictions
# Inputs: product_ids and horizon
# Returns a CSV
@st.cache_data(show_spinner="Making predictions...")
def make_predictions(ids, horizon):
    model_df = format_dataset(ids)
    sf = create_sf_object(model_df)
    forecast_df = sf.forecast(h=horizon)
    return forecast_df.to_csv(header=True)


if __name__ == "__main__":
    st.title("Forecast product demand")

    df = create_dataframe()

    st.subheader("Select a product")
    product_ids = st.multiselect(
        "Select product ID", options=df['parts_id'].unique())

    plot_volume(product_ids)

    with st.expander("Forecast"):
        if len(product_ids) == 0:
            st.warning("Select at least one product ID to forecast")
        else:
            horizon = st.slider("Horizon", 1, 12, step=1)

            forecast_btn = st.button("Forecast", type="primary")

            # Download CSV file if the forecast button is pressed
            if forecast_btn:
                csv_file = make_predictions(product_ids, horizon)
                # st.write(csv_file)
                st.download_button(
                    label="Download Prediction",
                    data=csv_file,
                    file_name="prediction.csv",
                    mime="text/csv"
                )
