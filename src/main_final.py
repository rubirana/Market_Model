""" Import packages and functions """

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from battery_arbitrage import run_three_stage_market
from Input_file import half_hourly_market_price
from output_file import output_to_excel_plots
#%%
""" Set case studies """
case = {}
case = {'Max charging rate' : 2,
        'Max discharging rate' : 2,
        'Max storage volume'          : 4,
        'Battery charging efficiency'         : 0.05,
        'Battery discharging efficiency'         : 0.05,
        'Lifetime (2)'         : 5000,      
        'Capex'              : 500000,
        'Fixed Operational Costs'  : 5000,      
          'start_year': 2018,
          'time_periods_per_day':48,
          'start_month' : 1 , 
          'start_day' : 1 , 
          'no_days':365,
          'Degradation_weight_market_3':5,
          'Degradation_weight_market_1_2':10,
          'minimum_energy_threshold':0.2
}
case['no_slots'] = case['no_days'] * case['time_periods_per_day']
market1_price_dict,market2_price_dict,market3_hf_price_dict=half_hourly_market_price(case)
daily_profit_market,daily_degradation_costs,Results_dict_market_2=run_three_stage_market(market1_price_dict,market2_price_dict,market3_hf_price_dict,case)
output_to_excel_plots(case,Results_dict_market_2,market1_price_dict,market2_price_dict,market3_hf_price_dict,daily_profit_market,daily_degradation_costs)

