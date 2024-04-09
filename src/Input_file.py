import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_data():
    excel_file_path = '../data/Second Round Technical Question - Attachment 2.xlsx'

    """ Loading data from excel sheet """
    Market_1_2 = pd.read_excel(excel_file_path, 
                               sheet_name='Half-hourly data', 
                               index_col=0, 
                               parse_dates=True)
    Market_3 = pd.read_excel(excel_file_path, 
                             sheet_name='Daily data', 
                             index_col=0, 
                             parse_dates=True)
    
    half_hourly_prices = Market_3.resample('30T').ffill()
    full_date_range = pd.date_range(start='2018-01-01', end='2020-12-31 23:30:00', freq='30T')
    Market_3_hf = half_hourly_prices.reindex(full_date_range, method='ffill')
    
    
    """ Extracting column names for market data """
    column_names = Market_1_2.columns
    column_names_list = list(column_names)
    
    """ Extracting market_1_2 data """
    market_price_columns = [col for col in column_names_list if "Price" in col]
    market_price_df = Market_1_2[market_price_columns]
    return market_price_df, Market_3_hf

def data_year(market_price_df,case,Market_3_hf):  
    start_date = pd.Timestamp(year=case['start_year'], month=case['start_month'], day=case['start_day'])
    duration_days = case['no_slots'] / 48
    end_date = start_date + pd.Timedelta(days=duration_days)
    market1_2_price_yearly = market_price_df[(market_price_df.index >= start_date) & 
                                         (market_price_df.index < end_date)]           
    market1_2_price_yearly.reset_index(drop=True, inplace=True)
    market1_2_price_subset = market1_2_price_yearly.iloc[:case['no_slots']]    
    market1_price_dict=market1_2_price_subset['Market 1 Price [£/MWh]']
    market2_price_dict=market1_2_price_subset['Market 2 Price [£/MWh]']   
    market3_hf_price_yearly= Market_3_hf[(Market_3_hf.index >= start_date) & 
                                         (Market_3_hf.index < end_date)] 
    market3_hf_price_yearly.reset_index(drop=True, inplace=True)
    market3_hf_price_subset = market3_hf_price_yearly.iloc[:case['no_slots']]    
    market3_hf_price_dict=market3_hf_price_subset['Market 3 Price [£/MWh]']
    return market1_price_dict ,market2_price_dict,market3_hf_price_dict

def half_hourly_market_price(case):
    market_price_df,Market_3_hf=read_data()
    market1_price_dict,market2_price_dict,market3_hf_price_dict =data_year(market_price_df,case,Market_3_hf)   
    return market1_price_dict,market2_price_dict,market3_hf_price_dict