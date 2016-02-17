function btsp(model, data)
% load data
switch(data)
    case 'Gilardoni2007'
        data = RepairableData('data/Gilardoni2007.txt');
    otherwise
        error('Data not found.')
end

% fit nonlinear model
switch(model)
    case 'plp'
        
    otherwise
        error('Model not implemented.')
end
end

function fval = model_plp(t, beta, theta)
lambda = @(t, beta, theta) (beta/theta) * (t/theta).^(beta - 1);
fval   = @(t)lambda(t,beta,theta); % expected number of failures
end