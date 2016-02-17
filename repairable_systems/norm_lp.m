function nrm = norm_lp(p,M,t,mcnf)
nrm = 0;

for i = 1:(length(t)-1)
    tmin  = t(i);
    tmax  = t(i+1);
    fun   = @(t)(abs(M(t) - 5).^p);
    nrm   = nrm + integral(fun,tmin,tmax);
end

nrm = nrm^(1/p);
fprintf('nrm%d(%f,%f) = %f\n', p, min(t), max(t), nrm);
end

function fx = step_func(x,t,y)
	index = sum(t <= x);
	fx    = y(index);
end