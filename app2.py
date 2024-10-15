import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="My App", page_icon="logo.JPG")
# st.header("This is a header with a divider", divider="gray")
tab1, tab2, tab3 = st.tabs(["Home","Health Index", "AboutUS"])
with tab1:
    from PIL import Image
    photo1 = Image.open("logo.jpg")
    st.image(photo1, width = 100)
    st.write("""
    ## Industrial Transformer Health Prediction(AI)
    """)
    from PIL import Image
    photo = Image.open("trans.jpeg")
    st.image(photo, width = 300)
    # uploaded_file = st.file_uploader("choose an image ...", type =  ['JPG', 'jpeg', 'png'])

    st.sidebar.image('data-entry-icon.png', width = 200, )
    st.sidebar.header('User Input Parameters')

    def user_input_features():
        Hydrogen = st.sidebar.slider('Hydrogen', 0, 25000, 500)
        Oxigen = st.sidebar.slider('Oxigen', 0, 250000, 100)
        Nitrogen = st.sidebar.slider('Nitrogen', 0, 100000, 40)
        Methane = st.sidebar.slider('Methane', 0, 10000, 1577)
        CO = st.sidebar.slider('CO', 0, 2000, 20)
        CO2 = st.sidebar.slider('CO2', 0, 30000, 20)
        Ethylene = st.sidebar.slider('Ethylene', 0, 20000, 100)
        Ethane = st.sidebar.slider('Ethane', 0, 6000, 40)
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
        col1, col2 = st.columns([1, 1])
        steps = [{'range': [0, 30], 'color': "Red"},
                {'range': [30, 50], 'color': "Orange"},
                {'range': [50, 70], 'color':"yellow"},
                {'range': [70, 85], 'color': "Lime"},
                {'range': [85, 100+1], 'color': 'Green'}
                ]
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

        st.markdown("""
            <style>
            .col-border {
                border: 2px solid #000;
                padding: 10px;
                margin: 10px;
            }
            </style>
        """, unsafe_allow_html=True)
        with col1:
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
        with col2:
            # st.markdown('<div class="col-border">Content in Column 2</div>', unsafe_allow_html=True)
            fig.update_layout(width = 300, height = 300)
            st.plotly_chart(fig)
        return prediction

    # trigger = st.button('Predict', on_click = predict)

    st.subheader('Prediction Probability')
    predict()

with tab2:
    photo3 = Image.open("expected-lifetime.png")
    st.image(photo3, width = 400)

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


## main
# import streamlit as st
# import pandas as pd
# from sklearn import datasets
# from sklearn.ensemble import RandomForestClassifier
# import joblib

# st.write("""
#     # AI Predictive Manintenance (APM)
# """)

# st.write("""
# # Transformer Health Prediction App
# This app predicts the **Transformer Health**!
# """)

# from PIL import Image
# photo = Image.open("iris.jpeg")
# st.image(photo)

# st.sidebar.header('User Input Parameters')

# def user_input_features():
#     sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.3)
#     sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
#     petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
#     petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
#     data = {'sepal_length': sepal_length,
#             'sepal_width': sepal_width,
#             'petal_length': petal_length,
#             'petal_width': petal_width}
#     features = pd.DataFrame(data, index=[0])
#     return features

# df = user_input_features()


# st.subheader('User Input parameters')
# st.write(df)

# # Machine Learning 
# # Started
# #------------------------------------------------
# iris = datasets.load_iris()
# X = iris.data
# Y = iris.target

# clf = RandomForestClassifier()
# clf.fit(X, Y)

# prediction = clf.predict(df)
# prediction_proba = clf.predict_proba(df)

# # Ended
# #-------------------------------------------------
# st.subheader('Class labels and their corresponding index number')
# st.write(iris.target_names)

# import time
# my_bar = st.progress(1)
# for percent_complete in range(100):
#     time.sleep(0.01)
#     my_bar.progress(percent_complete+1)

# st.subheader('Prediction Probability')
# st.write(prediction_proba)

# st.subheader('Prediction')
# result = iris.target_names[prediction]
# st.write(iris.target_names[prediction])
# #st.write(prediction)

# if result == 'setosa':
#     st.success("setosaaaa! win")
# elif result == 'versicolor':
#     st.success("versicolor! win")
# else:
#     st.success("virginica! win")
#     st.balloons()
