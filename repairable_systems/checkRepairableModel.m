clc;
close all;
% clear all;
%% read data and set model type
data  = RepairableData('data/Gilardoni2007.txt');

%% model parameters estimation
model_plp_fit   = RepairableModelPLP(data);
model_plp_mle   = RepairableModelPLP(data,'Algorithm','mle');
model_plp_cmle  = RepairableModelPLP(data,'Algorithm','cmle');
model_plp_bstrp = RepairableModelPLP(data,'Algorithm','bstrp');

%% plots
figure;
subplot(2,1,1);
data.plot_dots;
title('Plot original data');

% Create Mean Cumulative Number of Failures
subplot(2,1,2);
hold on; box on;
data.plot_mcnf;
model_plp_fit.plot('s');
model_plp_mle.plot('d');
model_plp_cmle.plot('x');
model_plp_bstrp.plot('o');
LEGEND = legend('show');
set(LEGEND,'Location','northwest');
