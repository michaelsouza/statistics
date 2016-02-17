clc; clear all;
addpath ./DERIVESTsuite

tau = @(b,t,c) t * (c * (b - 1))^(-1/b);
c = pi;
b = 1+rand;
t = rand;
f = @(y)tau(y(1),y(2),c);
g = grad_tau(c,b,t);
y = [b,t];
G = gradest(f,y);
fprintf('norm(d-g)/norm(g) = %E\n',norm(G-g)/norm(g));

N = randi(10);
T = rand(1,5);
x = rand;
y = [b,t];
loglike = @(b,t,x,N,T) N * log(b) - b * N * log(t) + (b - 1) * x - sum(T.^b)/t^b; 
f = @(y)loglike(y(1),y(2),x,N,T);
G = gradest(f,y);
g = grad_loglike(T,N,b,t,x);
fprintf('norm(d-g)/norm(g) = %E\n',norm(G-g)/norm(g));
h = hess_loglike(T,N,b,t);
[H,err] = hessian(f,y);   
fprintf('norm(H-h)/norm(h) = %E\n',norm(H-h)/norm(h));
disp(h)
disp(H)