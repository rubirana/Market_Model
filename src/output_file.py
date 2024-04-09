import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def output_to_excel_plots(case,Results_dict_market_2,market1_price_dict,market2_price_dict,market3_hf_price_dict,daily_profit_market,daily_degradation_costs):
    excel_file_path='../output/Results_Market_Analysis_2018.xlsx'

    start_datetime = f"{case['start_year']}-{case['start_month']:02d}-{case['start_day']:02d}"
    end_datetime = pd.to_datetime(start_datetime) + pd.Timedelta(days=case['no_days'])
    datetime_index = pd.date_range(start=start_datetime, end=end_datetime, periods=case['no_slots']+1, closed='left')
    
    
    df = pd.DataFrame(Results_dict_market_2, index=datetime_index)
    
   
    df['Total_Charging'] = df['Ch_market1'] + df['Ch_market2'] + df['Ch_market3']
    df['Total_Discharging'] = df['Dis_market1'] + df['Dis_market2'] + df['Dis_market3']
    
    yearly_profit = np.sum(daily_profit_market)
    yearly_degradation_cost=np.sum(daily_degradation_costs)
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
     df.to_excel(writer, sheet_name='Market Analysis', index=True)      
     yearly_summary = pd.DataFrame({
        'Yearly Profit': [np.sum(daily_profit_market)],
        'Yearly Degradation Cost': [np.sum(daily_degradation_costs)]
    })
     yearly_summary.to_excel(writer, sheet_name='Yearly Summary', index=False)
    
    plt.figure(figsize=(14, 20))
    # Market 3 HF Price
    plt.subplot(10, 1, 1)
    plt.plot(df.index, market3_hf_price_dict, label='Market 3 HF Price', color='black')
    plt.ylabel('Price [£/MWh]')
    plt.legend()
    # Market 3 HF Price
    plt.subplot(10, 1, 2)
    plt.plot(df.index, market1_price_dict, label='Market 1 HF Price', color='black')
    plt.ylabel('Price [£/MWh]')
    plt.legend()
    # Market 3 HF Price
    plt.subplot(10, 1, 3)
    plt.plot(df.index, market2_price_dict, label='Market 1 HF Price', color='black')
    plt.ylabel('Price [£/MWh]')
    plt.legend()
    # Charging Market1
    plt.subplot(10, 1, 4)
    plt.plot(df.index, Results_dict_market_2['Ch_market3'], label='Charging Market3', color='blue')
    plt.ylabel('Charging [MW]')
    plt.legend()
    
    # Discharging Market1
    plt.subplot(10, 1, 5)
    plt.plot(df.index, Results_dict_market_2['Dis_market3'], label='Discharging Market3', color='red')
    plt.ylabel('Discharging [MW]')
    plt.legend()
    plt.subplot(10, 1, 6)
    plt.plot(df.index, Results_dict_market_2['Ch_market1'], label='Charging Market1', color='blue')
    plt.ylabel('Charging [MW]')
    plt.legend()
    
    # Discharging Market1
    plt.subplot(10, 1, 7)
    plt.plot(df.index, Results_dict_market_2['Dis_market1'], label='Discharging Market1', color='red')
    plt.ylabel('Discharging [MW]')
    plt.legend()
    # Discharging Market1
    plt.subplot(10, 1, 8)
    plt.plot(df.index, Results_dict_market_2['Ch_market2'], label='Charging Market2', color='blue')
    plt.ylabel('Charging [MW]')
    plt.legend()
    # Discharging Market1
    plt.subplot(10, 1, 9)
    plt.plot(df.index, Results_dict_market_2['Dis_market2'], label='Discharging Market2', color='red')
    plt.ylabel('Discharging [MW]')
    
    # Charging Market2
    plt.subplot(10, 1, 10)
    plt.plot(df.index, Results_dict_market_2['SOC'], label='Energy', color='orange')
    plt.ylabel('Energy [MWh]')
    plt.legend()
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    cumulative_profit = np.cumsum(daily_profit_market)
    cumulative_degradation_cost=np.cumsum(daily_degradation_costs)
    
    plt.figure(figsize=(15, 10)) 
    plt.subplot(2, 1, 1)  # (rows, columns, panel number)
    plt.plot(cumulative_profit, label='Cumulative Market Profit', color='blue')
    
    plt.xlabel('Day of the Year')
    plt.ylabel('Cumulative Profit [£]')
    plt.legend()
    
    # Subplot 2 for Cumulative Degradation Cost
    plt.subplot(2, 1, 2)  # (rows, columns, panel number)
    plt.plot(cumulative_degradation_cost, label='Cumulative Degradation Cost', color='red')
    plt.xlabel('Day of the Year')
    plt.ylabel('Cumulative Degradation Cost [£]')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
        
        
