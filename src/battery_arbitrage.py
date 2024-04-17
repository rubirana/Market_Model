
import numpy as np
import pyomo.environ as pyo
from pyomo.environ import value
from pyomo.opt import SolverFactory
import pandas as pd

def run_battery_model_market_3(market1_price_dict,market2_price_dict,market3_hf_price_dict,case):   
    """
    Optimizes the operation of a battery for arbitrage in Market 3, considering the market's electricity prices
    and the battery's operational constraints. 
    Inputs:
       
        market3_hf_price_dict: Dict half hourly electricity prices in Market 3.
        case: Dict with model parameters including:
            - 'no_slots': Total number of time slots for the analysis.
            - 'Max charging rate': Maximum rate (MW) at which the battery can be charged.
            - 'Max discharging rate': Maximum rate (MW) at which the battery can be discharged.
            - 'Max storage volume': Maximum energy capacity of the battery (MWh).
            - 'Battery charging efficiency': Efficiency factor for charging.
            - 'Battery discharging efficiency': Efficiency factor for discharging.
            - 'minimum_energy_threshold': Minimum state of charge as a fraction of max storage.
            - 'Degradation_weight_market_3': Weight factor for degradation.
            - Other relevant parameters.

    Outputs:
        Results_Dict_market_3: Dict containing key results:
            - 'SOC': List of the battery's state of charge for each time slot.
            - 'Ch_market3': List of charging actions (MW) in Market 3 for each time slot.
            - 'Dis_market3': List of discharging actions (MW) in Market 3 for each time slot.
    The function sets up and solves an optimization problem using Pyomo and GLPK.
    """
    T = range(0, case['no_slots'])    
    model = pyo.ConcreteModel()    
    model.market3_hf_price= pyo.Param(T, initialize=market3_hf_price_dict)   
    model.p_ch_market3 = pyo.Var(T, within=pyo.NonNegativeReals)
    model.p_dis_market3 = pyo.Var(T, within=pyo.NonNegativeReals)    
    model.SOC = pyo.Var(T, within=pyo.NonNegativeReals,bounds=(case['minimum_energy_threshold']*case['Max storage volume'], case['Max storage volume']))
   ### These constraints will ensure charging and discharging power is constant whole day
    def constancy_discharge_rule(model, t): 
     if t % case['time_periods_per_day'] == 0:  
        return pyo.Constraint.Skip 
     return model.p_dis_market3[t] == model.p_dis_market3[t-1]
    model.constancy_discharge_constraint = pyo.Constraint(T, rule=constancy_discharge_rule)
    
    def constancy_charge_rule(model, t):
     if t % case['time_periods_per_day'] == 0:  
        return pyo.Constraint.Skip
     return model.p_ch_market3[t] == model.p_ch_market3[t-1]
    model.constancy_charge_constraint = pyo.Constraint(T, rule=constancy_charge_rule)
   
    # Constraints to limit charging and discharging power to maximum rates
    def market_com_ch(model,t):    
        return (  model.p_ch_market3[t] <= case['Max charging rate']  )
    model.mark1 = pyo.Constraint(T, rule=market_com_ch)
    
    def market_com_disch(model,t):     
        return ( model.p_dis_market3[t] <= case['Max discharging rate']  )
    model.mark2 = pyo.Constraint(T, rule=market_com_disch)    
    
    # SOC constraints       
    def soc(model, t):    
        if t == 0:
            return (model.SOC[t] == model.SOC[case['no_slots']-1] + (model.p_ch_market3[t])*(1-case['Battery charging efficiency']) - (model.p_dis_market3[t])/(1-case['Battery discharging efficiency']))
        else:
            return (model.SOC[t] == model.SOC[t-1] + (model.p_ch_market3[t])*(1-case['Battery charging efficiency']) - (model.p_dis_market3[t])/(1-case['Battery discharging efficiency']))
    model.soccon3a = pyo.Constraint(T, rule=soc)      
     
    def obj(model):
           """
           The objective function calculates net costs by subtracting revenue from discharging from the costs of charging and 
           adds a weighted term for degradation to incentivize minimal wear on the battery.
           """
           revenue_market3 = sum((model.p_dis_market3[t]*model.market3_hf_price[t]) for t in T)
           cost_market3 = sum((model.p_ch_market3[t] *  model.market3_hf_price[t]) for t in T)
           degradation = sum((model.p_ch_market3[t] +model.p_dis_market3[t]) for t in T)
                 
           return -revenue_market3 +cost_market3  +case['Degradation_weight_market_3']*degradation
    model.OBJ = pyo.Objective(rule=obj, sense=pyo.minimize)   
    opt = SolverFactory("glpk")  
    results = opt.solve(model, load_solutions=True,tee=True)  
    Results_Dict_market_3 = {
                     'SOC' : [value(model.SOC[t]) for t in T],
                    'Ch_market3' : [value(model.p_ch_market3[t]) for t in T],
                    'Dis_market3' : [value(model.p_dis_market3[t]) for t in T],
        }  
    return(Results_Dict_market_3)



def run_battery_model_market_1(market1_price_dict,market2_price_dict,market3_hf_price_dict,case,Results_Dict_market_3): 
    """
    Optimizes the operation of a battery for arbitrage in Market 1, considering the market's electricity prices
    and the battery's operational constraints and schedules from Market 3. 
    Inputs:
       
        market1_price_dict: Dict half hourly electricity prices in Market 1.
   
    Outputs:
        Results_Dict_market_1: Dict containing key results:
            - 'SOC': List of the battery's state of charge for each time slot.
            - 'Ch_market1': List of charging actions (MW) in Market 1 for each time slot.
            - 'Dis_market1': List of discharging actions (MW) in Market 1 for each time slot.
            - 'Ch_market3': List of charging actions (MW) in Market 3 for each time slot.
            - 'Dis_market3': List of discharging actions (MW) in Market 3 for each time slot.
    The function sets up and solves an optimization problem using Pyomo and GLPK.
    """
    T = range(0, case['no_slots']) 
  
    model = pyo.ConcreteModel()
    model.market1_price = pyo.Param(T, initialize=market1_price_dict)
    model.market2_price = pyo.Param(T, initialize=market2_price_dict)
    model.market3_hf_price= pyo.Param(T, initialize=market3_hf_price_dict)
    initial_ch_market3= {t: Results_Dict_market_3['Ch_market3'][t] for t in T}
    initial_dis_market3 = {t: Results_Dict_market_3['Dis_market3'][t] for t in T}
    model.p_ch_market3 = pyo.Param(T, initialize=initial_ch_market3)
    model.p_dis_market3 = pyo.Param(T, initialize=initial_dis_market3)   
    model.p_ch_market1 = pyo.Var(T, within=pyo.NonNegativeReals)
    model.p_dis_market1 = pyo.Var(T, within=pyo.NonNegativeReals)   
    model.SOC = pyo.Var(T, within=pyo.NonNegativeReals,bounds=(case['minimum_energy_threshold']*case['Max storage volume'], case['Max storage volume']))
    model.cumulative_charging = pyo.Var(T, within=pyo.NonNegativeReals)
    model.cumulative_discharging = pyo.Var(T, within=pyo.NonNegativeReals)
    model.cycle_count = pyo.Var(T, within=pyo.NonNegativeReals)
    ## Constraints to block the schedules from market 3
    def charging_market_3_blocks_discharging_market_1(model, t):      
        if initial_ch_market3[t] > 0:
            return (model.p_dis_market1[t] == 0)
        else:
            return pyo.Constraint.Skip
    model.charging_market_3_blocks_discharging_market_1_con = pyo.Constraint(T, rule=charging_market_3_blocks_discharging_market_1)

    def discharging_market_3_blocks_charging_market_1(model, t):      
        if initial_dis_market3[t] > 0:
            return (model.p_ch_market1[t] == 0)
        else:
            return pyo.Constraint.Skip 
    model.discharging_market_3_blocks_charging_market_1_con = pyo.Constraint(T, rule=discharging_market_3_blocks_charging_market_1)

    ## Constraints to ensure charging and discharging power in both the market together stays within the limit
    def market_1_3_inter_ch_1(model,t):      
        return (  model.p_ch_market1[t]+ model.p_ch_market3[t]  <= case['Max charging rate']  )
    model.market_inter_4 = pyo.Constraint(T, rule=market_1_3_inter_ch_1)
    
    def market_1_3_inter_dis_1(model,t):     
        return ( model.p_dis_market1[t]+ model.p_dis_market3[t]  <= case['Max discharging rate']  )
    model.market_inter_3 = pyo.Constraint(T, rule=market_1_3_inter_dis_1)    
     
    def soc_simple(model, t):
        if t == 0:
            return (model.SOC[t] == model.SOC[case['no_slots']-1] + (model.p_ch_market1[t] + model.p_ch_market3[t]) * (1 - case['Battery charging efficiency']) - (model.p_dis_market1[t] + model.p_dis_market3[t]) / (1 - case['Battery discharging efficiency']))
        else:
            return (model.SOC[t] == model.SOC[t-1] + (model.p_ch_market1[t] + model.p_ch_market3[t]) * (1 - case['Battery charging efficiency']) - (model.p_dis_market1[t] + model.p_dis_market3[t]) / (1 - case['Battery discharging efficiency']))
    model.soc_constraint = pyo.Constraint(T, rule=soc_simple)
 
   
    def obj(model):
          # Revenue, costs,degradation component from Market 1's operations
         
          revenue_market1 = sum((model.p_dis_market1[t] * model.market1_price[t]) for t in T)
          cost_market1 = sum((model.p_ch_market1[t] * model.market1_price[t]) for t in T)
          degradation = sum((model.p_ch_market1[t] +model.p_dis_market1[t]) for t in T)
         
          return -revenue_market1 + cost_market1 +case['Degradation_weight_market_1_2']* degradation 
         
    model.OBJ = pyo.Objective(rule=obj, sense=pyo.minimize)
    
    opt = SolverFactory("glpk")  
    results = opt.solve(model, load_solutions=True,tee=True)
   
   
    
    Results_dict_market_1 = {
    'SOC': [pyo.value(model.SOC[t]) for t in T],
    'Ch_market3': [pyo.value(model.p_ch_market3[t]) for t in T],
    'Dis_market3': [pyo.value(model.p_dis_market3[t]) for t in T],
    'Ch_market1': [pyo.value(model.p_ch_market1[t]) for t in T],
    'Dis_market1': [pyo.value(model.p_dis_market1[t]) for t in T],
     }

    return(Results_dict_market_1)


def run_battery_model_market_2(market1_price_dict,market2_price_dict,market3_hf_price_dict,case,Results_Dict_market_3,Results_dict_market_1): 
    """
    Optimizes the operation of a battery for arbitrage in Market 2, considering the market's electricity prices
    and the battery's operational constraints and schedules from Market 1 and 3. 
    Inputs:
       
        market2_price_dict: Dict half hourly electricity prices in Market 2.
   
    Outputs:
        Results_Dict_market_1: Dict containing key results:
            - 'SOC': List of the battery's state of charge for each time slot.
            - 'Ch_market3': List of charging actions (MW) in Market 3 for each time slot.
            - 'Dis_market3': List of discharging actions (MW) in Market 3 for each time slot.
            - 'Ch_market1': List of charging actions (MW) in Market 1 for each time slot.
            - 'Dis_market1': List of discharging actions (MW) in Market 1 for each time slot.
            - 'Ch_market2': List of charging actions (MW) in Market 2 for each time slot.
            - 'Dis_market2': List of discharging actions (MW) in Market 2 for each time slot.            
    The function sets up and solves an optimization problem using Pyomo and GLPK.
    """
    T = range(0, case['no_slots'])
    model = pyo.ConcreteModel()
    model.market1_price = pyo.Param(T, initialize=market1_price_dict)
    model.market2_price = pyo.Param(T, initialize=market2_price_dict)
    model.market3_hf_price = pyo.Param(T, initialize=market3_hf_price_dict)
    
    # Initialize parameters based on previous results
    initial_ch_market3 = {t: Results_Dict_market_3['Ch_market3'][t] for t in T}
    initial_dis_market3 = {t: Results_Dict_market_3['Dis_market3'][t] for t in T}
    initial_ch_market1 = {t: Results_dict_market_1['Ch_market1'][t] for t in T}
    initial_dis_market1 = {t: Results_dict_market_1['Dis_market1'][t] for t in T}
    
   # initialization
    model.p_ch_market3 = pyo.Param(T, initialize=initial_ch_market3)
    model.p_dis_market3 = pyo.Param(T, initialize=initial_dis_market3)
    model.p_ch_market1 = pyo.Param(T, initialize=initial_ch_market1)
    model.p_dis_market1 = pyo.Param(T, initialize=initial_dis_market1)
    
    # Market 2 variables
    model.p_ch_market2 = pyo.Var(T, within=pyo.NonNegativeReals)
    model.p_dis_market2 = pyo.Var(T, within=pyo.NonNegativeReals)
    model.SOC = pyo.Var(T, within=pyo.NonNegativeReals, bounds=(case['minimum_energy_threshold'] * case['Max storage volume'], case['Max storage volume']))
    model.is_charging = pyo.Var(T, within=pyo.Binary)
    model.is_discharging = pyo.Var(T, within=pyo.Binary)
    model.daily_cycles = pyo.Var(T, within=pyo.NonNegativeReals)


    def charging_in_market_1_or_3_blocks_discharging_market_2(model, t):
        # If there's charging in either Market 1 or Market 3 at time t, block discharging in Market 2
        if initial_ch_market1[t] > 0 or initial_ch_market3[t] > 0:
            return model.p_dis_market2[t] == 0
        else:
            return pyo.Constraint.Skip
    model.charging_in_market_1_or_3_blocks_discharging_market_2_con = pyo.Constraint(T, rule=charging_in_market_1_or_3_blocks_discharging_market_2)
    
    def discharging_in_market_1_or_3_blocks_charging_market_2(model, t):
        # If there's discharging in either Market 1 or Market 3 at time t, block charging in Market 2
        if initial_dis_market1[t] > 0 or initial_dis_market3[t] > 0:
            return model.p_ch_market2[t] == 0
        else:
            return pyo.Constraint.Skip
    model.discharging_in_market_1_or_3_blocks_charging_market_2_con = pyo.Constraint(T, rule=discharging_in_market_1_or_3_blocks_charging_market_2)

    ## These constraints ensures charging or discharging together in all the markets should stay within the limits
    def market_1_3_inter_ch_1(model,t):
       
        return (  model.p_ch_market1[t]+ model.p_ch_market3[t]+model.p_ch_market2[t]  <= case['Max charging rate']  )
    model.market_inter_4 = pyo.Constraint(T, rule=market_1_3_inter_ch_1)
    
    def market_1_3_inter_dis_1(model,t):
     
        return ( model.p_dis_market1[t]+ model.p_dis_market3[t]+model.p_dis_market2[t]  <= case['Max discharging rate']  )
    model.market_inter_3 = pyo.Constraint(T, rule=market_1_3_inter_dis_1)    
  

    def soc_simple(model, t):
        if t == 0:
            return (model.SOC[t] == model.SOC[case['no_slots']-1] + (model.p_ch_market1[t] + model.p_ch_market3[t]+model.p_ch_market2[t]) * (1 - case['Battery charging efficiency']) - (model.p_dis_market1[t] + model.p_dis_market3[t]+model.p_dis_market2[t]) / (1 - case['Battery discharging efficiency']))
        else:
            return (model.SOC[t] == model.SOC[t-1] + (model.p_ch_market1[t] + model.p_ch_market3[t]+model.p_ch_market2[t]) * (1 - case['Battery charging efficiency']) - (model.p_dis_market1[t] + model.p_dis_market3[t]+model.p_dis_market2[t]) / (1 - case['Battery discharging efficiency']))
    model.soc_constraint = pyo.Constraint(T, rule=soc_simple)
    def obj(model):
          revenue_market2 = sum(((model.p_dis_market2[t]) * model.market2_price[t]) for t in T)
          cost_market2 = sum((model.p_ch_market2[t])  * model.market2_price[t] for t in T)
          degradation = sum(model.p_ch_market2[t]+ model.p_dis_market2[t] for t in T)      
          return -1*revenue_market2+cost_market2 +case['Degradation_weight_market_1_2']*degradation 
  
    model.OBJ = pyo.Objective(rule=obj, sense=pyo.minimize)   
    opt = SolverFactory("glpk")  
    results = opt.solve(model, load_solutions=True,tee=True)    
    Results_dict_market_2 = {
    'SOC': [pyo.value(model.SOC[t]) for t in T],
    'Ch_market3': [pyo.value(model.p_ch_market3[t]) for t in T],
    'Dis_market3': [pyo.value(model.p_dis_market3[t]) for t in T],
    'Ch_market1': [pyo.value(model.p_ch_market1[t]) for t in T],
    'Dis_market1': [pyo.value(model.p_dis_market1[t]) for t in T],
    'Ch_market2': [pyo.value(model.p_ch_market2[t]) for t in T],
    'Dis_market2': [pyo.value(model.p_dis_market2[t]) for t in T],
     }
 

    return(Results_dict_market_2)


def calculate_daily_metrics(charge_list, discharge_list, price_dict,case):
    """
    Reorganising the time series charging and discharging scedules into daily format and calculating the daily cost and revenue
    """
    daily_profit = []
    daily_cost = []
    daily_revenue = []    
    days_in_year = len(charge_list) // case['time_periods_per_day']  # Assuming 48 half-hour intervals per day
    for day in range(days_in_year):
        daily_charge = charge_list[day*case['time_periods_per_day']:(day+1)*case['time_periods_per_day']]
        daily_discharge = discharge_list[day*case['time_periods_per_day']:(day+1)*case['time_periods_per_day']]
        daily_prices = [price_dict[t] for t in range(day*case['time_periods_per_day'], (day+1)*case['time_periods_per_day'])]       
        cost = sum(np.array(daily_charge) * np.array(daily_prices))
        revenue = sum(np.array(daily_discharge) * np.array(daily_prices))
        daily_cost.append(cost)
        daily_revenue.append(revenue)        
    return  daily_cost, daily_revenue





def calculate_daily_cycles_from_SOC(soc_data, case):
    """
    Calculate daily battery cycles based on SOC data.
     Inputs:   
          soc_data: List of SOC values for each time period.
          case: Dictionary containing 'Max storage volume' and other parameters.
     outputs: List of total battery cycles for each day.
    """
    
    daily_cycles = [0] * (len(soc_data) // case['time_periods_per_day'])
    cumulative_charge = 0
    cumulative_discharge = 0

    for i in range(1, len(soc_data)):
        soc_change = soc_data[i] - soc_data[i-1]

        if soc_change > 0:
            # Battery is charging
            cumulative_charge += soc_change
        else:
            # Battery is discharging
            cumulative_discharge -= soc_change  # soc_change is negative

        # Check if a full cycle has been completed
        if cumulative_charge >= case['Max storage volume'] and cumulative_discharge >= case['Max storage volume']:
            daily_index = i // case['time_periods_per_day']
            daily_cycles[daily_index] += 1
            # Reset cumulative counts for both charge and discharge
            cumulative_charge -= case['Max storage volume']
            cumulative_discharge -= case['Max storage volume']

        # At the end of each day, reset the cumulative discharge counter
        if i % case['time_periods_per_day'] == 0:
            cumulative_discharge = 0

    # Add the cycles from the last day if it's not a complete day
    if len(soc_data) % case['time_periods_per_day'] != 0:
        daily_cycles.append(1 if cumulative_charge >= case['Max storage volume'] and cumulative_discharge >= case['Max storage volume'] else 0)

    return daily_cycles




def calculate_battery_degradation_cost(results_dict,case):     
    daily_total_cycles = calculate_daily_cycles_from_SOC(results_dict['SOC'],  case )   
    cost_per_cycle =  case['Capex' ]  / case['Lifetime (2)']     
    daily_degradation_costs = [cost_per_cycle * cycles for cycles in daily_total_cycles]  
    return daily_degradation_costs

def calculate_profit(daily_total_cost_market,daily_total_revenue_market,case):
    daily_fixed_operational_cost=case['Fixed Operational Costs']/365
    daily_profit_market=daily_total_revenue_market-daily_total_cost_market-daily_fixed_operational_cost
    return daily_profit_market

    
def energy_arbritrage_cost_revenue(Results_dict_market_2,market1_price_dict,market2_price_dict,market3_hf_price_dict,case):

    daily_cost_market1, daily_revenue_market1 = calculate_daily_metrics(
        Results_dict_market_2['Ch_market1'],Results_dict_market_2['Dis_market1'],market1_price_dict,case)
    daily_cost_market2, daily_revenue_market2 = calculate_daily_metrics(
        Results_dict_market_2['Ch_market2'],
        Results_dict_market_2['Dis_market2'],
        market2_price_dict,
        case
    )
    daily_cost_market3, daily_revenue_market3 = calculate_daily_metrics(
        Results_dict_market_2['Ch_market3'],
        Results_dict_market_2['Dis_market3'],
        market3_hf_price_dict,
        case
    )
    # Calculate the daily total cost, revenue, and profit across all markets
    daily_total_cost_market = np.array(daily_cost_market1) + np.array(daily_cost_market2) + np.array(daily_cost_market3)
    daily_total_revenue_market = np.array(daily_revenue_market1) + np.array(daily_revenue_market2) + np.array(daily_revenue_market3)
    return daily_total_cost_market,daily_total_revenue_market





def run_three_stage_market(market1_price_dict,market2_price_dict,market3_hf_price_dict,case):
    """
     This  function provides optimal battery operation to maximize profit
     while managing degradation across three distinct markets.

    Inputs:
    - Half hourly market prices for all the three markets.
    - case (dict): A dictionary containing various operational parameters and settings for the model,
      such as 'no_slots', 'Max charging rate', 'Max discharging rate', etc.

    Process:
    1. Runs the battery arbitrage model for Market 3 to determine battery charging and discharging scedules.
    2. Runs the battery arbitrage model for Market 1, taking into account the results from Market 3.
    3. Runs the battery arbitrage model for Market 2, considering the outcomes from both Market 1 and Market 3.
    4. Calculates daily total costs and revenues from energy arbitrage across the markets.
    5. Estimates daily degradation costs of the battery based on its usage.
    6. Calculates daily profit by subtracting costs from revenues, including degradation costs.

    Returns:
    - daily_profit_market (list): Daily profit from energy arbitrage and battery operation.
    - daily_degradation_costs (list): Estimated daily degradation costs based on battery usage.
    - Results_dict_market_2 (dict): Final operational data including state of charge, charging, and
      discharging schedules for all markets.
   
    """
    Results_dict_market_3 = run_battery_model_market_3(market1_price_dict,market2_price_dict,market3_hf_price_dict,case)
    Results_dict_market_1 = run_battery_model_market_1(market1_price_dict,market2_price_dict,market3_hf_price_dict,case,Results_dict_market_3)
    Results_dict_market_2=  run_battery_model_market_2(market1_price_dict,market2_price_dict,market3_hf_price_dict,case,Results_dict_market_3,Results_dict_market_1) 
    daily_total_cost_market,daily_total_revenue_market=energy_arbritrage_cost_revenue(Results_dict_market_2,market1_price_dict,market2_price_dict,market3_hf_price_dict,case)
    daily_degradation_costs=calculate_battery_degradation_cost(Results_dict_market_2,case)
    daily_profit_market=calculate_profit(daily_total_cost_market,daily_total_revenue_market,case)
    return daily_profit_market,daily_degradation_costs,Results_dict_market_2