
# TO RUN THIS streamlit run app.py

import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))
st.markdown(
    f"""
    <div style='text-align:center'>
        <h1>Laptop Price Predictor</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# brand
company = st.selectbox('Brand Name',df['Company'].unique()) # GIves all the unique value in the dropdown which is inside the company column 

# type of laptop
type = st.selectbox('Type of Laptop',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Is it Touch Screen?',['No','Yes'])

# IPS
ips = st.selectbox('Does it have IPS Display?',['No','Yes'])

# screen size
screen_size = st.selectbox('What is the screen Size?',[10.1, 11.6, 12.1, 12.5, 13.3, 14.0, 15.0, 15.4, 15.6, 17.0, 17.3, 18.4, 21.5, 27.0])

# resolution
resolution = st.selectbox('What is the screen resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('Which Chip does it consists',df['Cpu brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('From which company is the Graphics Card from?',df['Gpu brand'].unique())

os = st.selectbox('What is the pre-installed Operating System?',df['os'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    st.title("Your Laptop is worth around: " + "Rs." + str(int(np.exp(pipe.predict(query)[0]))))# exp is done beause i had used log transformation in the model

