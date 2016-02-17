classdef RepairableModel < handle
    properties
        model_type = '';
        lambda     = [];
        p          = [];  % vector of parameters
    end
    methods
        function this = RepairableModel(mtype)
            this.model_type = mtype;
            
            switch mtype
                case 'plp'
                    this.lambda = @(t, p) (p(1)/p(2)) * (t/p(2)).^(p(1) - 1);
                    this.p = [1+rand,rand];
                otherwise
                    error('Not supported model.');
            end
        end
        
        function p = fit(this,data)
            fprintf('=== FIT =========================\n')

            t = data.mcnf.failureTimes;
            y = data.eval_mcnf(t);
            w = data.mcnf.numberOfUncensoredSystems;
            
            modelfun = @(p,t)this.eval_model(p,t);
            model  = fitnlm(t,y,modelfun,this.p,'Weights',w);
            this.p = model.Coefficients.Estimate;
            tau    = this.calc_tau(data);
            
            switch this.model_type
                case 'plp'
                    fprintf('> nlinfit(beta,theta)\n')
                    fprintf('  RMSE ........... % g\n', model.RMSE)
                    fprintf('  R2(ORD,ADJ) .... % g, % g \n', model.Rsquared.Ordinary, model.Rsquared.Adjusted);
                    fprintf('beta  ............ % g\n', this.p(1));
                    fprintf('theta ............ % g\n', this.p(2));
                    fprintf('tau .............. % g\n', tau);
                    
                    
                    %                     std_beta  = sqrt(CovB(1,1));
                    %                     std_theta = sqrt(CovB(2,2));
                    %                     fprintf('std(beta) ........ % g\n', std_beta)
                    %                     fprintf('std(theta) ....... % g\n', std_theta)
                    %                     fprintf('cor(beta,theta) .. % g\n', CovB(1,2)/(std_beta * std_theta))
            end
            p = this.p;
        end
        
        function [p,tau] = MLE(this,data)
            N = data.numberOfFailures; % total number of failures
            T = data.censorTimes;      % censor times
            t = data.failureTimes;     % get failures times
            
            fprintf('=== MLE (Analytical) ============\n');
            switch this.model_type
                case 'plp'
                    this.p(1) = RepairableModel.MLE_beta(t,T,N);
                    this.p(2) = RepairableModel.MLE_theta(this.p(1),T,N);
                    tau = this.calc_tau(data);
                    fprintf('beta ............. % g\n', this.p(1));
                    fprintf('theta ............ % g\n', this.p(2));
                    fprintf('tau .............. % g\n', tau);
                    
                    %                     T = data.censorTimes;
                    %                     N = data.numberOfFailures;
                    %                     H = hess_loglike(T,N,p(1),p(2));
                    %                     S = inv(-H);
                    %                     std_beta  = sqrt(S(1,1));
                    %                     std_theta = sqrt(S(2,2));
                    %                     fprintf('std(beta) ........ % g\n', std_beta);
                    %                     fprintf('std(theta) ....... % g\n', std_theta);
                    %                     fprintf('cor(beta,theta) .. % g\n', S(1,2) / (std_beta * std_theta));
            end
            p = this.p;
        end
        
        function [p,tau] = MLEC(this,data)
            M = RepairableModel.MLEC_M(data);
            
            fprintf('=== MLEC (Analytical) ===========\n');
            switch this.model_type
                case 'plp'
                    this.p(1) = RepairableModel.MLEC_beta(M,data);
                    this.p(2) = RepairableModel.MLEC_theta(this.p(1),M,data);
                    tau = this.calc_tau(data);
                    fprintf('beta ............. % g\n', this.p(1));
                    fprintf('theta ............ % g\n', this.p(2));
                    fprintf('tau .............. % g\n', tau);
                    
                    CI.max = chi2inv(0.975,2*M) * this.p(1) / (2*M);
                    CI.min = chi2inv(0.025,2*M) * this.p(1) / (2*M);
                    
                    %                     fprintf('CI95%%(beta) ......  [%g,%g]\n', CI.min, CI.max);
            end
            
            p = this.p;
        end
        
        function plot(this,p,data,dname,marker)
            tmax = max(data.censorTimes);
            t = 0:tmax/25:tmax;
            y = RepairableModel.eval_model(p,t,this.lambda);
            plot(t,y,'DisplayName',dname,'Marker',marker)
        end
    end
    
    methods(Access=private)
        function y = eval_model(this,p,t)
            y = zeros(size(t));
            for i = 1:length(t)
                y(i) = integral(@(t)this.lambda(t,p),0,t(i));
            end
        end
        
        function tau = calc_tau(this, data)
            eq_tau = @(tau)tau .* this.lambda(tau,this.p) - this.eval_model(this.p,tau) - 1/data.cost;
            fprintf('> fsolve(tau):%s\n',this.model_type);
            switch this.model_type
                case 'plp'
                    beta  = this.p(1);
                    theta = this.p(2);
                    tau = theta * ((beta - 1) * data.cost)^(-1/beta);
                    output.gap        = eq_tau(tau);
                    output.algorithm  = 'analytical';
                    output.iterations = 0;
                    %                     T = data.censorTimes;
                    %                     N = data.numberOfFailures;
                    %                     H = hess_loglike(T,N,beta,theta);
                    %                     g = grad_tau(data.cost,beta,theta);
                    %                     std_tau = sqrt(g * ((-H) \ g'));
                    %                     alpha = 0.05;
                    %                     cov_tau   = norminv(alpha/2) * std_tau;
                    %                     CI.min = tau - abs(cov_tau);
                    %                     CI.max = tau + abs(cov_tau);
                    %                     % upper confidence limit for H(\hat{tau}) - H(tau)
                    %                     Hucl = (std_tau^2 * beta * norminv(alpha/2)^2) / (2 * (tau)^3);
                otherwise
                    options = optimoptions('fsolve','Display','off');
                    tau    = max(data.censorTimes); % starting point
                    [tau,gap,~,output] = fsolve(eq_tau,tau,options);
                    output.gap = gap;
                    %             fprintf('tau .............. % g\n', tau);
                    %             fprintf('std(tau) ......... % g\n', std_tau);
                    %             fprintf('CI95%%(tau) ......   [%g, %g]\n', CI.min, CI.max);
                    %             fprintf('Hucl ............. % g\n', Hucl);
            end
            fprintf('  algorithm ......  %s\n', output.algorithm);
            fprintf('  gap ............ % g\n', output.gap);
            fprintf('  iterations ..... % d\n', output.iterations);
        end
    end
    
    methods(Access=private,Static)
        function beta = MLE_beta(t, T, N)
            options = optimoptions('fsolve','Display','off');
            [beta, gap, ~, output] = fsolve(@(beta)RepairableModel.MLE_eq_beta(beta, t, T, N), 1, options);
            fprintf('> fsolve(beta)\n');
            fprintf('  algorithm ......  %s\n', output.algorithm);
            fprintf('  gap ............ % g\n', gap);
            fprintf('  iterations ..... % d\n', output.iterations);
        end
        
        function theta = MLE_theta(beta, T, N)
            theta = sum(T.^beta / N)^(1/beta);
        end
        
        function gap = MLE_eq_beta(beta, t, T, N)
            y   = T.^beta;
            k   = N / sum(y);
            gap = N / (k * sum(y.* log(T)) - sum(log(t))) - beta;
        end
        
        function beta = MLEC_beta(M, data)
            k = 0;
            for i = 1:data.numberOfSystems
                ti = data.systems(i).failureTimes;
                Ti = data.systems(i).censorTime;
                k  = k + sum(log(Ti./ti));
            end
            beta = M / k;
        end
        
        function theta = MLEC_theta(beta,M,data)
            T = data.censorTimes;
            theta = sum(T.^beta / M)^(1/beta);
        end
        
        function M = MLEC_M(data)
            m = zeros(size(data.systems));
            for i = 1:data.numberOfSystems
                ti = data.systems(i).failureTimes;
                Ti = data.systems(i).censorTime;
                if(isempty(ti))
                    m(i) = 0;
                else
                    m(i) = length(ti) - (ti(end) == Ti);
                end
            end
            M = sum(m);
        end
    end
end