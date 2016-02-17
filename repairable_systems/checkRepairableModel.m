clc;
% clear all;
close all;
data  = RepairableData('data/Gilardoni2007.txt');
model = RepairableModel('plp');
p_fit  = model.fit(data);
p_mle  = model.MLE(data);
p_mlec = model.MLEC(data);

% figure; 
% subplot(2,1,1);
% data.plot_dots;
% subplot(2,1,2);
% hold on; box on;
% data.plot_mcnf;
% model.plot(p_fit ,data,'FIT','s');
% model.plot(p_mle ,data,'MLE','d');
% model.plot(p_mlec,data,'MLEC','x');
% legend('Location','NorthWest');