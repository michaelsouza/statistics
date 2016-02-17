classdef RepairableData < handle
    properties
        cost             = 0;  % CMR/CPM
        failureTimes     = []; % failure times
        censorTimes      = []; % censoring times
        numberOfFailures = 0;  % total number of failures
        numberOfSystems  = 0;
        systems          = [];
        
        % mcnf properties (arrays):
        % failureTimes: failure times (includes the last time censoring)
        % meanCumulativeNumberOfeFailures: mean cumulative number of failures
        % numberOfUncensoredSystems: number of uncensored systems 
        mcnf = struct('failureTimes',[],...
            'meanCumulativeNumberOfFailures',[],...
            'numberOfUncensoredSystems',[]);
    end
    
    methods
        function this = RepairableData(filename)
            db         = dlmread(filename);
            [nrows, ~] = size(db);
            ndata      = nrows - 1;
            
            this.cost  = db(1,1);   % the first entry is the ratio CMR/CPM
            this.systems = repmat(struct('failureTimes',[],'censorTime',0), ndata, 1);
            for i = 2:nrows
                % all times must be greater than zero
                index = db(i,:) > 0;
                dbi   = db(i,index); % data filtering
                % failure times
                this.systems(i-1).failureTimes = dbi(1:end-1);
                % the last entry of each row is the censoring time
                this.systems(i-1).censorTime   = dbi(end);
            end
            this.numberOfSystems  = length(this.systems);
            set_numberOfFailures(this);
            set_failureTimes(this);
            set_censorTimes(this);
            set_mcnf(this);
        end
        
        function fx = eval_mcnf(this,x)
            t    = this.mcnf.failureTimes;
            MCNF = this.mcnf.meanCumulativeNumberOfFailures;
            fx = zeros(size(x));
            for i = 1:length(x)
                if(x(i) > t(end))
                    error('The input parameter is out of the acceptable range.\n');
                end
                index = sum(t <= x(i));
                fx(i) = MCNF(index);
            end
        end
        
        function plot_mcnf(this)
            hold on; box on;
            t    = this.mcnf.failureTimes;
            MCNF = this.mcnf.meanCumulativeNumberOfFailures;
            x = zeros(4 * (length(t)-1), 1);
            y = zeros(size(x));
            j = 0;
            for i = 1:(length(t)-1)
                % plot([t(i),t(i+1)],[MCNF(i),MCNF(i)],'r.-');
                % plot([t(i+1),t(i+1)],[MCNF(i),MCNF(i+1)],'r-');
                x((j+1):(j+4)) = [t(i), t(i+1), t(i+1), t(i+1)];
                y((j+1):(j+4)) = [MCNF(i),MCNF(i),MCNF(i),MCNF(i+1)];
                j = j + 4;
            end
            plot(x,y,'.-','DisplayName','MCNF')
            xlim([0 max(this.censorTimes) * 1.05])
            title('Mean Cumulative Number of Failures');
            xlabel('time');
            ylabel('Number of Failures');
        end
        
        function plot_dots(this)
            hold on; box on;
            
            % dummy plot done only to set the legend
            plot(-1, -1,'-bx')
            plot(-1, -1,'-ro')
            legend('Censor time','Failure time','Location','Southwest');
            
            for i = 1:this.numberOfSystems
                plot([0,this.systems(i).censorTime], [i,i],'-bx')
                ti = this.systems(i).failureTimes;
                plot(ti, i * ones(size(ti)), 'ro');
            end
            ylabel('System','FontSize',12)
            xlabel('time','FontSize', 12)
            title(sprintf('Time dot plot (%d systems)', this.numberOfSystems) , 'FontSize', 15)
            ylim([0 (this.numberOfSystems + 1)])
            xlim([0 max(this.censorTimes) * 1.05])
        end
    end
    methods(Access=private)
        function set_mcnf(this)
            % Set the mean cumulative number of failures as a function of t (time).
            
            T = this.censorTimes;
            t_failures = sort(this.failureTimes);
            t = unique([0,t_failures,max(T)]);
            
            % mcnf(j): mean number of cumulative failures until time t(j)
            % q(j)   : number of uncensored systems until time t(j)
            MCNF = zeros(size(t));
            q    = zeros(size(t));
            for j = 1:length(t)
                q(j) = sum(T >= t(j));
                for i = 1:this.numberOfSystems
                    if(T(i) >= t(j))
                        MCNF(j) = MCNF(j) + sum(this.systems(i).failureTimes <= t(j));
                    end
                end
            end
            MCNF = MCNF./q; % adjust the average
            
            this.mcnf.failureTimes = t;
            this.mcnf.meanCumulativeNumberOfFailures = MCNF;
            this.mcnf.numberOfUncensoredSystems = q;
        end
        
        function set_censorTimes(this)
            % Returns an array with the censoring times of each system.
            this.censorTimes = [this.systems.censorTime];
        end
        
        function set_numberOfFailures(this)
            % Returns the total number of failures considering all systems.
            this.numberOfFailures = 0;
            for i = 1:this.numberOfSystems
                ti = this.systems(i).failureTimes;
                this.numberOfFailures = this.numberOfFailures + length(ti);
            end
        end
        
        function set_failureTimes(this)
            % Returns the sorted time of each failure considering all systems.
            % Notes:
            %  a) If two systems failed at the same time, the output will
            %  have duplicated entries;
            
            this.failureTimes = zeros(1,this.numberOfFailures);
            istart = 1;
            for i = 1:this.numberOfSystems
                ti = this.systems(i).failureTimes;
                ifinal = istart + length(ti) - 1;
                this.failureTimes(istart:ifinal) = ti;
                istart = ifinal + 1;
            end
            this.failureTimes = sort(this.failureTimes);
        end
    end
end