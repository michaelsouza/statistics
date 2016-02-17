clear all
syms a t b x y(b) N

% c = CMR / CPM
% t = theta
% b = beta
% y = sum_i^k T_i^b
% x = sum_ij log(t_ij)
% N = n_1 + n_2 + ... + n_k

tau = t * (a * (b - 1))^(-1/b);
v = [b,t];
g = jacobian(tau, v);
matlabFunction(g,'File','grad_tau');
s = latex(g);

% loglike = N * log(b) - b * N * log(t) + (b - 1) * x - y(b)/(t^b); 
% v = [b,t];
% g = jacobian(loglike,v);
% matlabFunction(g,'File','grad_loglike');
% h = jacobian(g,v);
% matlabFunction(h,'File','hess_loglike');