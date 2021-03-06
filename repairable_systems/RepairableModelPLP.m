classdef RepairableModelPLP < handle
    % Power Law Process (PLP)
    % lambda(t) = beta / theta * (t/theta)^(beta - 1)
    %
    % See the references at the end of this file.
    
    properties
        beta      = nan   % parameter of intensity function (lambda)
        theta     = nan   % parameter of intensity function (lambda)
        tau       = nan   % policy repair time interval
        H         = nan   % expected cost per unit of time
        ci        = struct('beta',[nan,nan],'theta',[nan,nan],'tau',[nan,nan],'H',[nan,nan]);
        verbose   = true  % print output (boolean)
        algorithm = 'fitnlm' % algorithm used to estimate plp parameters
        data      = [];
        type      = 'plp'
    end
    
    methods
        function this = RepairableModelPLP(data, varargin)
            % reading input parameters
            this.data = data;
            for i = 1:2:length(varargin)
                switch varargin{i}
                    case 'Algorithm'
                        % Set the algorithm used to estimate plp parameters
                        this.algorithm = varargin{i+1};
                    case 'Verbose'
                        this.verbose   = varargin{i+1};
                    otherwise
                        error('Input parameter %s is not supported.', varargin{i});
                end
            end
            
            % Estimates and set the parameters beta, theta
            switch this.algorithm
                case 'fitnlm'
                    % Nonlinear regression
                    this.fitnlm(data);
                case 'mle'
                    % Maximum Likelihood Estimator
                    this.MLE(data);
                case 'cmle'
                    % Conditioned Maximum Likelihood Estimator
                    this.CMLE(data);
                case 'bstrp'
                    % Bootstrap
                    this.bstrp(data);
                otherwise
                    error('Not supported algorithm')
            end
            
            % calculate and set the optimal time interval tau
            this.calc_tau(data);
            
            % calculate and set H(tau)
            this.ExpectedCostPerUnitOfTime(data);
            
            % display results
            if(this.verbose)
                fprintf('beta  ............ % 9.3g [% 9.3g, % 9.3g]\n', this.beta, this.ci.beta);
                fprintf('theta ............ % 9.3g [% 9.3g, % 9.3g]\n', this.theta, this.ci.theta);
                fprintf('tau .............. % 9.3g [% 9.3g, % 9.3g]\n', this.tau, this.ci.tau);
                fprintf('H ................ % 9.3g [% 9.3g, % 9.3g]\n', this.H, this.ci.H);
                fprintf('L1 distance ...... % 9.3g\n', data.distance(@(t)this.ExpectedNumberOfFailures(t), 'L1'));
                fprintf('L2 distance ...... % 9.3g\n', data.distance(@(t)this.ExpectedNumberOfFailures(t), 'L2'));
            end
        end
        
        function y = lambda(this,t,beta,theta)
            % Evaluates the intensity function.
            % t     :: time
            % beta  :: (optional)
            % theta :: (optional)
            
            if(nargin < 3)
                beta  = this.beta;
                theta = this.theta;
            end
            y = (beta/theta) * (t/theta).^(beta - 1);
        end
        
        function Nt = ExpectedNumberOfFailures(this,t,beta,theta)
            % Calculates N(t), the expected number of failures until time t.
            % beta  :: plp parameter (optional)
            % theta :: plp parameter (optional)
            if(nargin < 3)
                beta  = this.beta;
                theta = this.theta;
            end
            % Model evaluation - Expected number of failures until time t.
            Nt = (t./theta).^beta;
        end
        
        function plot(this, marker)
            tmax = max(this.data.censorTimes);
            t = 0:tmax/25:tmax;
            y = this.ExpectedNumberOfFailures(t);
            plot(t,y,'DisplayName',['PLP:' this.algorithm],'Marker',marker)
        end
        
        %% TAU
        function calc_tau(this, data)
            % See [Gilardoni2007] eq. (5) pp. 50
            this.tau = this.theta * (data.CPM / ((this.beta - 1) * data.CMR))^(1/this.beta);
            
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
            
            if(this.verbose)
                fprintf('> fsolve(tau)\n');
                fprintf('  algorithm ......  %s\n', 'analytical');
                fprintf('  gap ............ % g\n', this.check_tau());
                fprintf('  iterations ..... % d\n', 0);
            end
        end
        function gap = check_tau(this)
            % Check the error (gap) in the current tau value.
            % See [Gilardoni2007] eq. (4) pp. 49
            gap = this.tau .* this.lambda(this.tau) - this.ExpectedNumberOfFailures(this.tau) - this.data.CPM/this.data.CMR;
        end
        
        %% H(tau) :: Expected cost per unit of time
        function ExpectedCostPerUnitOfTime(this, data)
            % Expected cost per unit of time
            % See [Gilardoni2007] eq.(2) pp. 49
            this.H = (data.CPM + data.CMR * this.ExpectedNumberOfFailures(this.tau)) / this.tau;
        end
        
        %% BOOTSTRP
        function bstrp(this, data)
            
            % saving original verbose value
            vb = this.verbose;
            
            % turn off prints
            this.verbose = false;
            
            % call bootstrap
            options = statset('UseParallel',true);
            nboot   = 10000;
            N = data.numberOfSystems;
            x = 1:N; % id of each system
            [CI,bootstat] = bootci(nboot,{@(x)this.bstrp_fun(data, x),x},'Options',options);
            
            % set estimatives
            this.beta  = mean(bootstat(:,1));
            this.theta = mean(bootstat(:,2));
            this.tau   = mean(bootstat(:,3));
            this.H     = mean(bootstat(:,4));
            
            % set confidence intervals
            this.ci.beta  = CI(:,1);
            this.ci.theta = CI(:,2);
            this.ci.tau   = CI(:,3);
            this.ci.H     = CI(:,4);
            
            % restoring verbose original value
            this.verbose = vb;
            
            if(this.verbose)
                fprintf('=== PLP:BOOTSTRAP ================\n')
                
                figure;
                
                % plot beta
                subplot(2,4,1); hold on; box on;
                histogram(bootstat(:,1),50,'Normalization','probability');
                plot(this.beta,0,'o');
                xlabel('beta');
                ylabel('p(beta)');
                title('BSTRP :: Histogram - BETA')
                
                subplot(2,4,5); hold on; box on;
                [f,x] = ksdensity(bootstat(:,1)); % estimate density (normal)
                plot(x,f); plot(this.beta,0,'o');
                xlabel('beta');
                title('BSTRP :: Density - BETA')
                
                % plot theta
                subplot(2,4,2); hold on; box on;
                histogram(bootstat(:,2),50,'Normalization','probability');
                plot(this.theta,0,'o');
                xlabel('theta');
                ylabel('p(theta)');
                title('BSTRP :: Histogram - THETA')
                
                subplot(2,4,6); hold on; box on;
                [f,x] = ksdensity(bootstat(:,2)); % estimate density (normal)
                plot(x,f); plot(this.theta,0,'o');
                xlabel('theta');
                title('BSTRP :: Density - THETA');
                
                % plot tau
                subplot(2,4,3); hold on; box on;
                histogram(bootstat(:,3),50,'Normalization','probability');
                plot(this.tau,0,'o');
                xlabel('tau');
                ylabel('p(tau)');
                title('BSTRP :: Histogram - TAU')
                
                subplot(2,4,7); hold on; box on;
                [f,x] = ksdensity(bootstat(:,3)); % estimate density (normal)
                plot(x,f); plot(this.tau,0,'o');
                xlabel('tau');
                title('BSTRP :: Density - TAU');
                
                % plot H
                subplot(2,4,4); hold on; box on;
                histogram(bootstat(:,4),50,'Normalization','probability');
                plot(this.H,0,'o');
                xlabel('H(tau)');
                ylabel('p(H(tau))');
                title('BSTRP :: Histogram - H(tau)')
                
                subplot(2,4,8); hold on; box on;
                [f,x] = ksdensity(bootstat(:,4)); % estimate density (normal)
                plot(x,f); plot(this.H,0,'o');
                xlabel('H(tau)');
                title('BSTRP :: Density - H(tau)');
            end
        end
        function p = bstrp_fun(this, data, x)
            % x : id of the systems to be used
            
            % setup new data (d)
            d.CMR             = data.CMR;
            d.CPM             = data.CPM;
            d.systems         = data.systems(x);
            d.numberOfSystems = length(d.systems);
            
            % set censor times
            d.censorTimes = [d.systems.censorTime];
            
            % number of failures
            d.numberOfFailures = 0;
            for i = 1:d.numberOfSystems
                d.numberOfFailures = d.numberOfFailures + length(d.systems(i).failureTimes);
            end
            
            % failure times
            d.failureTimes = sort([d.systems.failureTimes]);
            
            % set beta and theta using CMLE
            this.CMLE(d);
            
            % set tau = tau(beta,theta)
            this.calc_tau(d);
            
            % set ECT = H(tau) (See eq.(2) [Gilardoni2007] pp. 49)
            this.ExpectedCostPerUnitOfTime(d);
            
            % set output
            p = [this.beta, this.theta, this.tau, this.H];
        end
        
        %% FITNLM :: Nonlinear Regression
        function fitnlm(this,data)
            % Estimates the intensity function (rho) parameters by fitting
            % the data Mean Cumulative Number Of Failures (mcnf).
            
            t = data.mcnf.failureTimes;
            y = data.eval_mcnf(t);
            w = data.mcnf.numberOfUncensoredSystems;
            
            % initial random values
            p = [1+rand,rand]; % p = [beta,theta];
            
            % nonlinear regression (fit)
            modelfun = @(p,t)this.ExpectedNumberOfFailures(t,p(1),p(2));
            model    = fitnlm(t,y,modelfun,p,'Weights',w);
            p        = model.Coefficients.Estimate;
            CI       = model.coefCI;
            
            % set estimators
            this.beta  = p(1);
            this.theta = p(2);
            
            % set confidence intervals
            this.ci.beta  = CI(1,:);
            this.ci.theta = CI(2,:);
            
            if(this.verbose)
                fprintf('=== PLP:FITNLM ===================\n')
                fprintf('> nlinfit(beta,theta)\n')
                fprintf('  RMSE ........... % g\n', model.RMSE)
                fprintf('  R2(ORD,ADJ) .... % g, % g \n', model.Rsquared.Ordinary, model.Rsquared.Adjusted);
            end
        end
        
        %% MLE :: Maximum Likelihood Estimator
        function MLE(this,data)
            N = data.numberOfFailures; % total number of failures
            T = data.censorTimes;      % censor times
            t = data.failureTimes;     % get failures times
            
            this.beta  = this.MLE_beta(t,T,N);
            this.theta = this.MLE_theta(T,N);
            
            T   = data.censorTimes;
            N   = data.numberOfFailures;
            HLL = this.MLE_H(T,N,this.beta,this.theta); % Hess(log(L(beta,theta)))
            S   = inv(-HLL);
            std_beta  = sqrt(S(1,1));
            std_theta = sqrt(S(2,2));
            fprintf('  std(beta) ...... % g\n', std_beta);
            fprintf('  std(theta) ..... % g\n', std_theta);
            fprintf('  cor(beta,theta). % g\n', S(1,2) / (std_beta * std_theta));
        end
        function beta = MLE_beta(this, t, T, N)
            % Maximum Likelihood Estimator (MLE)
            % See [Rigdon2000] section 5.4 pp. 207.
            options = optimoptions('fsolve','Display','off');
            [beta, gap, ~, output] = fsolve(@(beta)this.MLE_check_beta(beta, t, T, N), 1, options);
            if(this.verbose)
                fprintf('=== PLP:MLE ======================\n');
                fprintf('> fsolve(beta)\n');
                fprintf('  algorithm ......  %s\n', output.algorithm);
                fprintf('  gap ............ % g\n', gap);
                fprintf('  iterations ..... % d\n', output.iterations);
            end
        end
        function gap = MLE_check_beta(~,beta, t, T, N)
            % Check the error on beta value.
            % See [Rigdon2000] eq. (5.13) pp. 209.
            y   = T.^beta;
            k   = N / sum(y);
            gap = N / (k * sum(y.* log(T)) - sum(log(t))) - beta;
        end
        function theta = MLE_theta(this,T, N)
            % Calculates the parameter theta using beta current value.
            % See [Rigdon2000] eq. (5.11) pp. 207
            theta = sum(T.^this.beta / N)^(1/this.beta);
        end
        function h = MLE_H(~,T,N,beta,theta)
            % Log Likelihood Hessian
            %    This function was generated by the Symbolic Math Toolbox version 6.2.
            
            t2  = theta.^(-beta);
            t3  = sum(T.^beta);
            t4  = log(theta);
            t5  = sum((T.^beta).*log(T));
            t6  = 1.0./theta;
            t7  = -beta-1.0;
            t8  = theta.^t7;
            t9  = beta.*t5.*t8;
            t10 = sum((T.^beta).*((log(T)).^2));
            h = reshape([-N.*1.0./beta.^2-t2.*t10-t2.*t3.*t4.^2+t2.*t4.*t5.*2.0,t9-N.*t6+t3.*t8-beta.*t3.*t4.*t8,t9-N.*t6+t2.*t3.*t6-beta.*t3.*t4.*t8,N.*beta.*1.0./theta.^2-beta.*theta.^(-beta-2.0).*t3.*(beta+1.0)],[2, 2]);
        end
        
        %% CMLE :: Conditional Maximum Likelihood Estimator
        function CMLE(this,data)
            % See [Ringdon2000] pp. 210 and [Crow1975]
            
            M = this.CMLE_M(data);
            this.beta  = this.CMLE_beta(M,data);
            this.theta = this.CMLE_theta(this.beta,M,data);
            
            % See [Rigdon2000] pp. 211
            CI.max = chi2inv(0.975,2*M) * this.beta / (2*M);
            CI.min = chi2inv(0.025,2*M) * this.beta / (2*M);
            
            this.ci.beta = [CI.min, CI.max];
            if(this.verbose)
                fprintf('=== PLP:CMLE (Analytical) ========\n');
            end
        end
        function M = CMLE_M(~,data)
            % See [Ringdon2000] pp. 210
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
        function beta  = CMLE_beta(~,M,data)
            % Conditional MLE.
            % See [Ringdon2000] pp. 210
            k = 0;
            for i = 1:data.numberOfSystems
                ti = data.systems(i).failureTimes;
                Ti = data.systems(i).censorTime;
                k  = k + sum(log(Ti./ti));
            end
            beta = M / k;
        end
        function theta = CMLE_theta(~,beta,M,data)
            % See [Ringdon2000] pp. 210
            T     = data.censorTimes;
            theta = sum(T.^beta / M)^(1/beta);
        end
    end
end

% References:
%
% @book{Rigdon2000,
%   title={Statistical methods for the reliability of repairable systems},
%   author={Rigdon, Steven E and Basu, Asit P},
%   year={2000},
%   publisher={Wiley New York}
% }
%
% @article{Gilardoni2007,
%   title={Optimal maintenance time for repairable systems},
%   author={Gilardoni, Gustavo L and Colosimo, Enrico A},
%   journal={Journal of Quality Technology},
%   volume={39},
%   number={1},
%   pages={48--53},
%   year={2007},
%   publisher={[Milwaukee]: American Society for Quality Control.}
% }
%
% @techreport{Crow1975,
%   title={Reliability analysis for complex, repairable systems},
%   author={Crow, Larry H},
%   year={1975},
%   institution={DTIC Document}
% }
