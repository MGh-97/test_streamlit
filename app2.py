import streamlit as st
import pycaret
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Fanavaran AI", page_icon="logo.JPG")
# Predefined username and password (for simplicity)
USERNAME = "admin"
PASSWORD = "password123"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def logout():
    st.session_state.authenticated = False

# Authentication function
def authenticate(username, password):
    if username == USERNAME and password == PASSWORD:
        st.session_state.authenticated = True
        st.success("Login successful!")
    else:
        st.error("Invalid username or password.")

# Application Interface
if st.session_state.authenticated:

    tab1, tab2, tab3 = st.tabs(["Home","Health Index information", "AboutUS"])

    with tab1:
        from PIL import Image
        photo = Image.open("logo.jpg")
        st.image(photo, width = 100)
        st.write("""
        ## Industrial Transformer Health Prediction(AI)
        """)
        col1, col2 = st.columns([1, 1])
        with col1:
            photo1 = Image.open("R.jpeg")
            st.image(photo1, width=300)
        with col2:
            photo2 = Image.open("thermography-inspection-transformator.jpg")
            st.image(photo2, width= 300)
        # uploaded_file = st.file_uploader("choose an image ...", type =  ['JPG', 'jpeg', 'png'])
        st.sidebar.image('data-entry-icon.png', width = 200)
        st.sidebar.header('User Input Parameters')

        def user_input_features():
            Hydrogen = st.sidebar.slider('Hydrogen(ppm)', 0, 25000, 500)
            Oxigen = st.sidebar.slider('Oxigen(ppm)', 0, 250000, 100)
            Nitrogen = st.sidebar.slider('Nitrogen(ppm)', 0, 100000, 40)
            Methane = st.sidebar.slider('Methane(ppm)', 0, 10000, 1577)
            CO = st.sidebar.slider('CO(ppm)', 0, 2000, 20)
            CO2 = st.sidebar.slider('CO2(ppm)', 0, 30000, 20)
            Ethylene = st.sidebar.slider('Ethylene(ppm)', 0, 20000, 100)
            Ethane = st.sidebar.slider('Ethane(ppm)', 0, 6000, 40)
            Acethylene = st.sidebar.slider('Acethylene', 0, 10000, 50)
            DBDS = st.sidebar.slider('DBDS', 0, 300, 100)
            Power_factor = st.sidebar.slider('Power_factor', 0, 100, 2)
            InterfacialV = st.sidebar.slider('InterfacialV', 0, 60, 10)
            Dielectric_rigidity = st.sidebar.slider('Dielectric_rigidity', 0, 100, 10)
            water_content = st.sidebar.slider('water_content', 0, 200, 0)
            data = {'Hydrogen': Hydrogen,
                    'Oxigen': Oxigen,
                    'Nitrogen': Nitrogen,
                    'Methane': Methane,
                    'co': CO,
                    'co2': CO2,
                    'Ethylene': Ethylene,
                    'Ethane': Ethane,
                    'Acethylene': Acethylene,
                    'DBDS': DBDS, 
                    'Power factor': Power_factor,
                    'Interfacial V': InterfacialV,
                    'Dielect ricrigidity': Dielectric_rigidity,
                    'Water content': water_content}
            features = pd.DataFrame(data, index=[0])
            return features
        columns = [
                'Hydrogen','Oxigen','Nitrogen','Methane',
                'CO','CO2','Ethylene','Ethane','Acethylene',
                'DBDS','Power factor','Interfacial V',
                'Dielectric rigidity','Water content'
                ]
        df = user_input_features()
        st.subheader('AI Predictive Manintenance..user input parameters')
        st.write(df)
        st.bar_chart(df, horizontal = True)

        model = joblib.load('Trans-Health.pkl')

        def predict():
            row = df.values.flatten()
            X = pd.DataFrame([row], columns=columns)
            prediction = model.predict(X)[0]
            fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = prediction,
                            title = {'text': "Trasformer Health Index (%)"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "Crimson"},
                                'steps' : [
                                    {'range': [0, 30], 'color': "Red"},
                                    {'range': [30, 50], 'color': "Orange"},
                                    {'range': [50, 70], 'color':"yellow"},
                                    {'range': [70, 85], 'color': "Lime"},
                                    {'range': [85, 100+1], 'color': 'Green'}]
                            }))

            col3, col4 = st.columns([1, 1])
            with col3:
                # st.markdown('<div class="col-border">Content in Column 1</div>', unsafe_allow_html=True)
                st.write("\n"*5)
                st.write("\n"*5)
                st.write("\n"*5)
                st.write("\n"*5)
                st.write("\n"*5)
                if 85<=prediction<=100:
                    st.success(" Very Good ----- Expected Lifetime : More than 15 years.........Requirements : Normal Maintenance")
                elif 70<=prediction <85:
                    st.success("good.......Expected Liffetime: More than 10 years Normal.......Requirement :  Maintenance")
                elif 50<= prediction< 70:
                    st.warning('Fair ......Expected Lifetime :From 3-10 years.......Requirements: Increase diagnostic testing, possible remedial work or replacement needed depending on criticality')
                elif 30<= prediction < 50:
                    st.error("poor.......Expected Lifetime : Less than 3 year........Requirements : Start planning process to replace or rebuild considering risk and consequences of failure")
                elif 0<=prediction<30:
                    st.error("Very Poor........Expected Lifetime : Near to the end of life..........Requirements : Immediately assess risk; replace or rebuild based on assessment")
            with col4:
                # st.markdown('<div class="col-border">Content in Column 2</div>', unsafe_allow_html=True)
                fig.update_layout(width = 300, height = 300)
                st.plotly_chart(fig)
            return prediction

        # trigger = st.button('Predict', on_click = predict)
        st.subheader('Prediction Probability')
        predict()

    with tab2:
        photo2 = Image.open("expected-lifetime.png")
        st.image(photo2, width = 400)

    with tab3:
        # About us
        st.title("About Us")
        # st.image("your_logo.png", width=200)  # Replace with your logo's path

        st.write("""
        Welcome to our company! We specialize in providing top-notch AI solutions that drive innovation and efficiency.
        Our team is dedicated to delivering high-quality products and services to meet your unique needs.
        """)

        st.write("""
        **Our Mission**
        To revolutionize industries with advanced AI technologies, creating solutions that enhance productivity and foster growth.

        **Our Vision**
        To be the leading provider of AI-driven solutions, empowering businesses worldwide with cutting-edge technology.

        **Our Values**
        - Innovation
        - Integrity
        - Customer Satisfaction
        - Excellence
        """)
        # st.image("team_photo.jpg")  # Replace with your team photo's path
        st.write("""
                **Contact Us :**

                Website : www.fanavaran-sharif.com

                Email: info@fanavaran-sharif.com

                Phone: +982188742844
        """)
        photo3 = Image.open("logo.jpg")
        st.image(photo3, width = 100)

    if st.button("Logout"):
        logout()

else:
    st.title("**AI Predictiva Maintenance App**")
    # st.subheader("Please enter your UserName & Password:")
    
    # Input fields for username and password
    input_username = st.text_input("Username")
    input_password = st.text_input("Password", type="password")
    
    # Login button
    if st.button("Login"):
        authenticate(input_username, input_password)
        # st.experimental_rerun()
