import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import numpy as np 
import matplotlib.pyplot as plt 
import math 
import seaborn as sns

def pc_curve(Site,Month_number,Year):
    # 10.194.112.9
    username = 'chakradhar'
    password = 'Chakri%401881'
    host = '10.194.112.9:1433'
    database_name = 'mytrahprod'
    table_name = 'raw_myWind_imput_WTG'

    engine = create_engine(f'mysql+mysqlconnector://{username}:{password}@{host}/{database_name}')

    connection = engine.connect()
    
    Hub_height = {
    'Sinner': 1027,
    'Aspari II':544,
    'Aspari 2':544,
    'Aspari I':576,
    'Burgula':623,
    'Chakla':480.5,
    'Jamanwada':116,
    'Mahidad':365,
    'Kaladonger':264,
    'Maniyachi':187,
    'Mokal':305,
    'Vagarai':358.41,
    'Nidhi':295,
    'Nipaniya':543.56,
    'Nazeerabad':793,
    'Pottipadu':550,
    'Bhesada':440,
    'Savalsung':688,
    'Vajrakarur':584
    }
    ref_csv_file = {
        'Sinner':'Sinner_ref.csv' ,
        'Aspari II':'Aspari-2_ref.csv',
        'Aspari 2':'Aspari-2_ref.csv',
        'Aspari I':'Aspari-1_ref.csv',
        'Burgula':'Burugula_ref.csv',
        'Chakla':'Chakla_ref.csv',
        'Jamanwada':'Jamanwada_ref.csv',
        'Mahidad':'Mahidad_ref.csv',
        'Kaladonger':'Kaladonger_ref.csv',
        'Maniyachi':'Maniyachi_ref.csv',
        'Mokal':'Mokal_ref.csv',
        'Vagarai':'Vagarai_ref.csv',
        'Nidhi':'Nidhi_ref.csv',
        'Nipaniya':'Nipaniya_ref.csv',
        'Nazeerabad':'Nazeerabad_ref.csv',
        'Pottipadu':'Pottipadu_ref.csv',
        'Bhesada':'Beshada_ref.csv',
        'Savalsung':'Savalsung_ref.csv',
        'Vajrakarur':'Vajrakarur_ref.csv'
    }
    month_dict={
        'JAN':1,
        'FEB':2,
        'MAR':3,
        'APR':4,
        'MAY':5,
        'JUNE':6,
        'JULY':7,
        'AUG':8,
        'SEPT':9,
        'OCT':10,
        'NOV':11,
        'DEC':12
        
    }
    site_name=Site
    month=month_dict[f'{Month_number}']
    month=int(month)
    year=Year
    year=int(year)
    
    query = f"SELECT * FROM {database_name}.{table_name} WHERE Site='{site_name}';"
    df = pd.read_sql_query(query, connection)

    connection.close()

    df_ref = pd.read_csv(ref_csv_file[f'{site_name}'])
    df_ref = df_ref.dropna()

    hub_elev = Hub_height[f'{site_name}']
    std_air_density=1.225
    
    df=df[['Timestamp','LocNo','Site','Temperature (◦C)','Wind speed(m/s)','Active Power (kW)']]

    machine_names = df['LocNo'].unique()

    df['Year'] = df['Timestamp'].dt.year 
    df['Month'] = df['Timestamp'].dt.month
    df = df[(df['Month'] == month) & (df['Year'] == year)]
    
    df = df.rename(columns={'LocNo':'Machine_name','Active Power (kW)':'Active_power','Wind speed(m/s)':'wind_speed','Temperature (◦C)':'Temp_Outdoor'})

    df['Temp_kelvin']=df['Temp_Outdoor']+273.15
    df['Air_density_site']= (353.05/df['Temp_kelvin'])* (np.exp((-0.034*hub_elev)/df['Temp_kelvin']))
    df['Density_correction']=np.power(df['Air_density_site']/std_air_density,1/3)
    df['Corrected_wind_speed']=df['wind_speed']*df['Density_correction']

    # creating x1 and x2
    df['x1'] = np.floor(df['Corrected_wind_speed'])
    df['x2'] = np.ceil(df['Corrected_wind_speed'])

    #creating y1
    df = df.merge(df_ref, left_on='x1', right_on='Windspeed (m/s)', how='left')
    df = df.drop('Windspeed (m/s)', axis=1)
    df = df.rename(columns={'Power (KW)':'y1'})

    #creating y2
    df = df.merge(df_ref, left_on='x2', right_on='Windspeed (m/s)', how='left')
    df = df.drop('Windspeed (m/s)', axis=1)
    df = df.rename(columns={'Power (KW)':'y2'})

    #calculating Expected power 
    df['Expected_power']=(((df['y2']-df['y1'])*(df['Corrected_wind_speed']-df['x1']))/(df['x2']-df['x1']))+df['y1']

    df['Condition'] = 'Green'  # Default color
    df.loc[df['Active_power'] < df['Expected_power'], 'Condition'] = 'Red'


    #Expected generation active time 
    df['Expected_power_active_time'] = np.where(df['Active_power'] > 0, df['Expected_power'], 0)

    #Break down loss
    df['Break_down_loss']=np.where(df['Active_power'] > 0,0,df['Expected_power'])

    #difference in KWH
    df['Diff(kwh)']=np.where(df['Active_power'] > 0,(df['Expected_power']-df['Active_power'])/6,0)


    # calculating units for particular machine and adding in new data frame 
    df_labels1= df[df['Active_power'] > 0].groupby('Machine_name')['Active_power'].sum().reset_index()
    df_labels1['Active_power']=df_labels1['Active_power']/6

    df_labels2= df[df['Expected_power'] > 0].groupby('Machine_name')['Expected_power'].sum().reset_index()
    df_labels2['Expected_power']=df_labels2['Expected_power']/6

    df_labels3= df[df['Expected_power_active_time'] > 0].groupby('Machine_name')['Expected_power_active_time'].sum().reset_index()
    df_labels3['Expected_power_active_time']=df_labels3['Expected_power_active_time']/6
    

    df_labels4= df[df['Break_down_loss'] > 0].groupby('Machine_name')['Break_down_loss'].count().reset_index()
    df_labels4['Break_down_loss']=df_labels4['Break_down_loss']/6
    


    df_labels = df_labels1.merge(df_labels2, on='Machine_name',how='left')

    df_labels = df_labels.merge(df_labels3, on='Machine_name', how='left')

    df_labels = df_labels.merge(df_labels4, on='Machine_name', how='left')

    df_labels = df_labels.rename(columns={'Active_power':'Actual_Generation_Kwh','Expected_power':'Total_Expected_Generation_Kwh','Expected_power_active_time':'Expected_Generation_Active_time_Kwh','Break_down_loss':'Breakdown_hours'})


    df_labels['Break_down_loss']=df_labels['Total_Expected_Generation_Kwh']-df_labels['Expected_Generation_Active_time_Kwh']
    df_labels['PC_loss/gain']=df_labels['Actual_Generation_Kwh']-df_labels['Expected_Generation_Active_time_Kwh']
    df_labels['Deviation_%']=(df_labels['PC_loss/gain']/df_labels['Expected_Generation_Active_time_Kwh'])*100
    df_labels['PR_%']=(df_labels['Actual_Generation_Kwh']/df_labels['Expected_Generation_Active_time_Kwh'])*100

    df_labels = df_labels.round(1)
    
    st.header('SUMMARY :')
    st.write(df_labels)
    
    st.header('POWER CURVES :')
    # Define the number of subplots per page
    subplots_per_page = 6

    # Split the list of machine_names into chunks of size subplots_per_page
    chunks = [machine_names[i:i+subplots_per_page] for i in range(0, len(machine_names), subplots_per_page)]

    for chunk in chunks:
        # Create a new page with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        fig.suptitle(f"{site_name}", fontsize=16)

        for Machine_name, ax in zip(chunk, axes.ravel()):
            subset = df[df['Machine_name'] == Machine_name]
            mask = df_labels['Machine_name'] == Machine_name
            if mask.any():  # Check if there are matching rows in df_labels
                x = df_labels.loc[mask, 'Break_down_loss'].values[0]
                y = df_labels.loc[mask, 'PC_loss/gain'].values[0]
                z = df_labels.loc[mask, 'Deviation_%'].values[0]
                a = df_labels.loc[mask, 'PR_%'].values[0]
                b = df_labels.loc[mask, 'Breakdown_hours'].values[0]

                text = f"Breakdown loss    : {x:.1f} Kwh\nBreakdown hours : {b:.1f} hours\nPC loss/gain          : {y:.1f} Kwh\nDeviation              : {z:.1f} %\nPR                         : {a:.1f} %"
            else:
                # Handle the case when there are no matching rows
                text = f"Active power is not recorded\n for this WTG in {Month_number}-{Year} month "
            
            ax.set_title(f'Machine: {Machine_name}', fontsize=12)
            ax.set_xlabel('Wind_Speed(m/s)', fontsize=9)
            ax.set_ylabel('Active_Power(Kw)', fontsize=9)
            scatter = sns.scatterplot(data=subset, x='Corrected_wind_speed', y='Active_power', hue='Condition',
                                palette={'Red': '#FF4040', 'Green': '#31661a'}, ax=ax)
            scatter.legend().set_visible(False)
            ax.plot(df_ref['Windspeed (m/s)'], df_ref['Power (KW)'], color='#000080', linewidth=2.5)
            ax.text(12.5, 100, text, fontsize=7)
            ax.set_xlim([0, 20])


            # loop for removing extra 
        for i in range(len(chunk), subplots_per_page):
            axes.ravel()[i].axis('off')
            
        st.pyplot(fig)
    
    

def main():
    
    
    st.title('Generate Power Curve Analysis')
    
    Site = st.selectbox('Select_Site', ('Sinner','Aspari II','Aspari 2','Aspari I','Burgula','Burgula GE','Burugla GE','Chakla','Jamanwada','Mahidad','Kaladonger','Maniyachi','Mokal','Vagarai','Nidhi','Nipaniya','Nazeerabad','Pottipadu','Bhesada','Savalsung','Vajrakarur'))
    Month_number=st.selectbox('Select_Month',('JAN','FEB','MAR','APR','MAY','JUNE','JULY','AUG','SEPT','OCT','NOV','DEC'))
    Year=st.selectbox('Select_Year',(2023,2024,2025))
    
    if st.button('Generate Report'):
        pc_curve(Site,Month_number,Year)


if __name__=='__main__':
    main()
