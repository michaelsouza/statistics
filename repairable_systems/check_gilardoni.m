clear all
theta = 24844;
beta  = 1.988;
cost  = 15;

tau   = theta * (cost * (beta-1)) ^(-1/beta);

fprintf('tau = %f\n', tau)

std_beta  = 0.401;
std_theta = 2973.1;
cor = -0.34;
cov = cor * std_beta * std_theta;

g = grad_tau(cost, beta, theta);

S = [std_beta^2 cov;cov std_theta^2];

fprintf('std(tau) = %f\n', sqrt(g*S*g'))