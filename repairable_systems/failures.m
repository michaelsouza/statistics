function [x,y,X] = failures()    
    S = {};
    S{  end+1   }=[	0	8839	17057	21887	];
    S{	end+1	}=[	0	9280	16442	21887	];
    S{	end+1	}=[	0	10445	13533		];
    S{	end+1	}=[	0	7902			];
    S{	end+1	}=[	0	8414			];
    S{	end+1	}=[	0	13331			];
    S{	end+1	}=[	0	17156	21887		];
    S{	end+1	}=[	0	16305	21887		];
    S{	end+1	}=[	0	16802	21887		];
    S{	end+1	}=[	0	4881			];
    S{	end+1	}=[	0	16625			];
    S{	end+1	}=[	0	7396	7541	19590	];
    S{	end+1	}=[	0	1821			];
    S{	end+1	}=[	0	15821	19746	19877	];
    S{	end+1	}=[	0	1927			];
    S{	end+1	}=[	0	15813	21886		];
    S{	end+1	}=[	0	15524	21886		];
    S{	end+1	}=[	0	21440			];
    S{	end+1	}=[	0	369			];
    S{	end+1	}=[	0	11664	17031	21857	];
    S{	end+1	}=[	0	7544			];
    S{	end+1	}=[	0	6039			];
    S{	end+1	}=[	0	2168	6698		];
    S{	end+1	}=[	0	18840	21879		];
    S{	end+1	}=[	0	2288			];
    S{	end+1	}=[	0	2499			];
    S{	end+1	}=[	0	17100			];
    S{	end+1	}=[	0	10668	16838		];
    S{	end+1	}=[	0	15550	21887		];
    S{	end+1	}=[	0	1616			];
    S{	end+1	}=[	0	14041	20004		];
    S{	end+1	}=[	0	21888			];
    S{	end+1	}=[	0	21888			];
    S{	end+1	}=[	0	21888			];
    S{	end+1	}=[	0	21888			];
    S{	end+1	}=[	0	21888			];
    S{	end+1	}=[	0	21888			];
    S{	end+1	}=[	0	21888			];
    S{	end+1	}=[	0	21888			];
    S{	end+1	}=[	0	21888			];
    S{	end+1	}=[	0	21888			];

    %% create X = struct(x,[],y[])
    nfailures = 0;
    n = length(S);
    X = repmat(struct('x',[],'y',[]),n,1);
    xmax = 0;
    for i = 1:n
        xi = S{i};
        yi = zeros(size(xi));
        ni = length(xi);
        for j = 2:(ni - 1)
            yi(j:end) = yi(j:end) + 1;
            xmax = max(xmax, xi(ni - 1)); % time of the last failure
        end    
        X(i).x = xi;
        X(i).y = yi;        
        nfailures = nfailures + ni - 2;
    end
    fprintf('Total number of failures is %d.\n', nfailures);

    %% event times
    xticks = unique(cell2mat(S));

    X     = create_curves(X);
    [x,y] = calc_mean(X,xticks);

    close all
    fig = figure;
    [xmax,xlims] = plot_dots(fig,X);
    plot_mean(x,y,xmax,xlims);
end

function X = create_curves(X)
n = length(X);
for i = 1:n
    xi = X(i).x;
    yi = X(i).y;
    ni = length(xi);
    x = zeros(2 * ni - 1, 1);
    y = x;
    x(1) = xi(1);
    y(1) = yi(1);
    for j = 1:(ni-1)
        x(2 * j)     = xi(j + 1);
        y(2 * j)     = yi(j);
        x(2 * j + 1) = xi(j + 1);
        y(2 * j + 1) = yi(j + 1);
    end        
    X(i).curve.x = x;
    X(i).curve.y = y;
end
end

function plot_mean(x,y,xmax,xlims)
    sbplt = subplot(2,1,2);
    box(sbplt,'on'); hold(sbplt,'all');
    n = length(x);    
    for i = 1:(n-1)
        plot([x(i),x(i+1)],[y(i),y(i)],'Parent',sbplt,'LineWidth',2,'Color','k','Marker','none')
        plot(x(i)  ,y(i),'Parent',sbplt,'LineWidth',2,'Color','k','Marker','o','MarkerFaceColor','k')
        plot(x(i+1),y(i),'Parent',sbplt,'LineWidth',2,'Color','k','Marker','o')
    end
    plot([x(end),xmax],[y(end),y(end)],'Parent',sbplt,'LineWidth',2,'Color','k','Marker','none')
    plot(x(end),y(end),'Parent',sbplt,'LineWidth',2,'Color','k','Marker','o','MarkerFaceColor','k')
    plot(xmax  ,y(end),'Parent',sbplt,'LineWidth',2,'Color','k','Marker','o')
    xlabel('Failures and PMs times (hours)')
    ylabel('Mean cumulative number of failures');
    title('Mean cumulative number of failures versus time');
    xlim(xlims);
end

function [xmax,xlims] = plot_dots(fig,X)
    sbplt = subplot(2,1,1,'Parent',fig,'YDir','reverse');
    box(sbplt,'on'); hold(sbplt,'all');
    n = length(X);
    xmax = 0;
    for i = 1:n
        x = X(i).x;
        y = ones(size(x)) * i;
        xmax = max([x,xmax]);
        plot(x, y, 'Parent', sbplt, 'Marker', '.', 'Color', 'k', 'MarkerSize', 15);
        plot([x(1), x(end)], [y(1), y(end)], 'Parent', sbplt, 'LineStyle','none','Marker', '.', 'Color', [1 0 0], 'MarkerSize', 15);
    end
    xlims = [0 1.05 * xmax];
    ylim([0 (n + 1)])
    xlim(xlims);
    xlabel('Failures and PMs times (hours)');
    ylabel('System');
    title('Time dot plot for all systems unit');
end

function [x,y] = calc_mean(X,xticks)
    n = length(X);
    x = xticks;
    y = zeros(size(x));
    for i = 1:n
        Xx = X(i).x;
        Xy = X(i).y;
        for j = 1:length(x)
            xj = x(j);
            yj = feval_step(Xx,Xy,xj);
            y(j) = y(j) + yj;
        end
    end

    % normalization
    y = y / n;

    % eliminating redundant points
    index = false(size(x));
    ys = unique(y);
    for i = 1:length(ys)
        ifrst = find(ismember(y,ys(i)),1,'first');
        index(ifrst) = true;        
    end
    y = y(index);
    x = x(index);
end

function fx = feval_step(Xx,Xy,x)
    k  = sum(Xx <= x);
    fx = Xy(k);
end