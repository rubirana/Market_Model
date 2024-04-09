clc
clear

tic  % To read the time elapsed
T = 1:96; % Assuming 48 half-hourly slots for a day.
M = 5000; % Big M value for binary constraints.
plot_week = 10; % Week to show in plots of  charging/discharging and prices.

% Battery parameters
Max_storage = 4; % Maximum storage volume in MW
Max_charging_rate = 2; % Maximum charging rate in MW
Max_discharging_rate = 2; % Maximum discharging rate in MW
Battery_charging_efficiency = 0.95; % Charging efficiency
Battery_discharging_efficiency = 0.95; % Discharging efficiency


opts = detectImportOptions('Second Round Technical Question - Attachment 2.xlsx', 'Sheet', 'Half-hourly data', 'VariableNamingRule', 'preserve');
opts.SelectedVariableNames = {'Market 1 Price [£/MWh]', 'Market 2 Price [£/MWh]'};
halfHourlyData = readtable('Second Round Technical Question - Attachment 2.xlsx', opts); halfHourlyData_list = table2array(halfHourlyData);
opts = detectImportOptions('Second Round Technical Question - Attachment 2.xlsx', 'Sheet', 'Daily data', 'VariableNamingRule', 'preserve');
opts.SelectedVariableNames = {'Market 3 Price [£/MWh]'};
dailyData = readtable('Second Round Technical Question - Attachment 2.xlsx', opts); dailyData_list = table2array(dailyData);
halfHourlyDay_list = repelem(dailyData_list, 48, 1);


SOC = sdpvar(length(T), 1, 'full'); 
p_ch_market1 = sdpvar(length(T), 1, 'full'); p_dis_market1 = sdpvar(length(T), 1, 'full');
p_ch_market2 = sdpvar(length(T), 1, 'full'); p_dis_market2 = sdpvar(length(T), 1, 'full');
p_ch_market3 = sdpvar(length(T), 1, 'full'); p_dis_market3 = sdpvar(length(T), 1, 'full');
y_charge = binvar(length(T), 1, 'full'); % Binary variable for charging
y_discharge = binvar(length(T), 1, 'full'); % Binary variable for discharging
% partial_cycles = sdpvar(1, 'full');
% cycle_counter = intvar(1, 'full');

Constraints = [];
for t = T
    Constraints = [Constraints, sum([p_ch_market1(t), p_ch_market2(t), p_ch_market3(t)]) <= y_charge(t) * M];
    Constraints = [Constraints, sum([p_dis_market1(t), p_dis_market2(t), p_dis_market3(t)]) <= y_discharge(t) * M];
   
    Constraints = [Constraints, y_charge(t) + y_discharge(t) <= 1];
    Constraints = [Constraints, sum([p_ch_market1(t), p_ch_market2(t), p_ch_market3(t)]) <= Max_charging_rate];
    Constraints = [Constraints, sum([p_dis_market1(t), p_dis_market2(t), p_dis_market3(t)]) <= Max_discharging_rate];
    
    if t == 1
        Constraints = [Constraints, SOC(t) == 4]; % Initial SOC assumed 4 MWh.
    else
        Constraints = [Constraints, SOC(t) == SOC(t-1) + (p_ch_market1(t) + p_ch_market2(t) + p_ch_market3(t)) * Battery_charging_efficiency - (p_dis_market1(t) + p_dis_market2(t) + p_dis_market3(t)) / Battery_discharging_efficiency];
    end
    Constraints = [Constraints, 0 <= SOC(t) <= Max_storage]; 
    
end

Constraints = [Constraints, p_ch_market1 >= 0];
Constraints = [Constraints, p_dis_market1 >= 0];
Constraints = [Constraints, p_ch_market2 >= 0];
Constraints = [Constraints, p_dis_market2 >= 0];
Constraints = [Constraints, p_ch_market3 >= 0];
Constraints = [Constraints, p_dis_market3 >= 0];

market1_price=halfHourlyData_list(T,1);  market2_price=halfHourlyData_list(T,2); market3_price=halfHourlyDay_list(T);

Objective = sum((p_ch_market1 .* market1_price) - (p_dis_market1 .* market1_price)) + sum((p_ch_market2 .* market2_price) - (p_dis_market2 .* market2_price)) + sum((p_ch_market3 .* market3_price) - (p_dis_market3 .* market3_price));

options = sdpsettings('solver', 'gurobi');
sol = optimize(Constraints, Objective, options);

toc  % End of time track.

SOC_value = value(SOC);

p_ch_market1_value = value(p_ch_market1);
p_dis_market1_value = value(p_dis_market1);
p_ch_market2_value = value(p_ch_market2);
p_dis_market2_value = value(p_dis_market2);
p_ch_market3_value = value(p_ch_market3);
p_dis_market3_value = value(p_dis_market3);

sum_charging =   p_ch_market1_value + p_ch_market2_value + p_ch_market3_value;
sum_discharging = p_dis_market1_value + p_dis_market2_value + p_dis_market3_value;


% figure % To check any violation of charging discharging per time slot. 
% subplot(2,1,1)
% plot(sum_charging)
% ylabel('All Market Charging')
% subplot(2,1,2)
% plot(sum_discharging)
% ylabel('All Market Discharging')



figure
t = T;
subplot(3,2,1)
plot(t, p_ch_market1_value, 'b', 'LineWidth', 2)
title('Market 1 Charging')
xlabel('Time Slot')
ylabel('MW')
subplot(3,2,2)
plot(t, p_dis_market1_value, 'r', 'LineWidth', 2)
title('Market 1 Discharging')
xlabel('Time Slot')
ylabel('MW')
subplot(3,2,3)
plot(t, p_ch_market2_value, 'b', 'LineWidth', 2)
title('Market 2 Charging')
xlabel('Time Slot')
ylabel('MW')
subplot(3,2,4)
plot(t, p_dis_market2_value, 'r', 'LineWidth', 2)
title('Market 2 Discharging')
xlabel('Time Slot')
ylabel('MW')
subplot(3,2,5)
plot(t, p_ch_market3_value, 'b', 'LineWidth', 2)
title('Market 3 Charging')
xlabel('Time Slot')
ylabel('MW')
subplot(3,2,6)
plot(t, p_dis_market3_value, 'r', 'LineWidth', 2)
title('Market 3 Discharging')
xlabel('Time Slot')
ylabel('MW')



figure

subplot(3,1,1)
plot(t, market1_price, 'k', 'LineWidth', 2)
title('Market 1 Prices')
xlabel('Time Slot')
ylabel('Price')
subplot(3,1,2)
plot(t, market2_price, 'k', 'LineWidth', 2)
title('Market 2 Prices')
xlabel('Time Slot')
ylabel('Price')
subplot(3,1,3)
plot(t, market3_price, 'k', 'LineWidth', 2)
title('Market 3 Prices')
xlabel('Time Slot')
ylabel('Price')


figure
plot(t, SOC_value, '-')
title('SOC of BESS')
xlabel('Time Slot')
ylabel('SOC [MWh]')
