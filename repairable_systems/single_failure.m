function single_failure()
clc;
data = RepairableData('data/Gilardoni2007.txt');

[beta,theta] = MLE(data);

close all;
subplot(2,2,1);
data.plot_dots
subplot(2,2,3);
data.plot_mcnf
subplot(2,2,2)
plot_fit(beta,theta,data);
set(gcf,'units','normalized','outerposition',[0.4801    0.0533    0.1788    0.9000])

MLEC(data);
end

function nrm = norm_lp(p,M,t,mcnf)
nrm = 0;
for i = 1:(length(t)-1)
    fun = @(t)abs(M(t) - mcnf(i)).^p;
    nrm = nrm + integral(fun,t(i),t(i+1));
end
nrm = nrm^(1/p);
end

function tau = calc_tau(model, data, beta, theta)
switch model
    case 'power_law'
        tau = theta * ((beta - 1) * data.cost)^(-1/beta);
        g   = grad_tau(data.cost,beta,theta);
        T   = data.censorTimes;
        N   = data.numberOfFailures;
        H   = hess_loglike(T,N,beta,theta);
        std_tau = sqrt(g * ((-H) \ g'));
        alpha = 0.05;
        cov   = norminv(alpha/2) * std_tau;
        CI.min = tau - abs(cov);
        CI.max = tau + abs(cov);
        % upper confidence limit for H(\hat{tau}) - H(tau)
        Hucl = (std_tau^2 * beta * norminv(alpha/2)^2) / (2 * (tau)^3);
    otherwise
        error('Not implemented');
end
S = inv(-H);
std_beta  = sqrt(S(1,1));
std_theta = sqrt(S(2,2));
fprintf('model = %s\n', model);
fprintf('tau   = %g\n', tau);
fprintf('Hucl  = %g\n', Hucl);
fprintf('std(beta)        is %g\n', std_beta);
fprintf('std(theta)       is %g\n', std_theta);
fprintf('cor(beta,theta)  is %g\n', S(1,2) / (std_beta * std_theta));
fprintf('std(tau)         is %g\n', std_tau);
fprintf('CI(95%%) for tau  is [%g,%g]\n', CI.min, CI.max);

fprintf('=== Conclusion from Gilardoni2007\n');
fprintf('The company would lose at most %3.2f x CPM monetary units per \n', 365 * 24 * Hucl);
fprintf('year per equipment by doing %3.2f hours compared with the \n', tau);
fprintf('true optimal policy if perfect information about beta and \n');
fprintf('theta were available.\n');
end

function plot_fit(beta, theta, data)
lambda = @(t, beta, theta) (beta/theta) * (t/theta).^(beta - 1);
T      = data.censorTimes;

% plot adjusted function
M  = @(t)lambda(t,beta,theta); % expected number of failures
dt = max(T) / 100;
t  = 0:dt:max(T);
Mt = zeros(size(t));
for i = 1:length(t)
    Mt(i) = integral(M, 0, t(i));
end

hold on; box on;
plot(t, Mt,'LineWidth', 2); % plot fit
data.plot_mcnf;             % plot MCNF
xlabel('time','FontSize',12);
ylabel('Number of Failures','FontSize',12);
title('Goodness-of-fit analysis', 'FontSize', 15)
xlim([0 max(T) * 1.05])
legend('M(t)','MCNF','Location','Northwest')

% fprintf('=== Goodness-of-fit analysis\n');
% fprintf('distance L1 is %f\n', norm_lp(1,M,t,mcnf))
% fprintf('distance L2 is %f\n', norm_lp(2,M,t,mcnf))
end


