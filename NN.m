classdef NN
    properties(SetAccess = private)
    % Network static parameters
        Layers %--> Array of network layers
        num_labels %--> Number of output labels

    % Network current dynamic parameters
        J %--> Network cost value
        Grad %--> Network gradient
 
    % Network current input features
        X %--> Features

    % Training data
        lambda %--> Training regularization constant
        max_iter %--> Maximum number of iterations

    % Network diagnostics
        J_train %--> Network training cost
        J_cv %--> Network cross-validation cost
        Acc_cv %--> Network cross-validation accuracy
    end
    methods
        function obj = NN(dim, activ, num) %--> Neural network constructor
            % Check neural network dimensions are correct
            if(size(dim, 2) < 2 || any(dim <= 0) || ~isinteger(dim))
                error('neural network architecture is incorrect!!');
            else
                dim = double(dim);
            end

            % Fill the network with layers, activation units and weights
            obj.Layers = Layer.empty(0, size(dim, 2));
            grad_len = 0; %--> Unrolled gradient length
            for i = 1 : size(dim, 2) - 1
                obj.Layers(i) = Layer(activ(i), dim(i), dim(i + 1));
                grad_len = grad_len + size(get_grad(obj.Layers(i)), 1) *...
                    size(get_grad(obj.Layers(i)), 2);
            end
            obj.Layers(i + 1) = Layer(activ(i + 1), dim(i + 1), 0);

            % Determine number of labels
            obj.num_labels = num;

            % Initialize cost and gradient
            obj.J = 0;
            obj.Grad = zeros(grad_len, 1);
        end
        function p = get_pred(obj) %--> Get prediction "Hypothesis"
            p = get_a(obj.Layers(end));
        end
        function l = get_lambda(obj) %--> Get regularization
            l = obj.lambda;
        end
        function j = get_train_cost(obj) %--> Get training cost
            j = obj.J_train;
        end
        function j = get_cv_cost(obj) %--> Get cross-validation cost
            j = obj.J_cv;
        end
        function a = get_acc(obj) %--> Get classification accuracy
            a = obj.Acc_cv;
        end
        function obj = set_lambda(obj, lmbd) %--> Set regularization
            obj.lambda = lmbd;
        end
        function obj = set_data(obj, x) %--> Set training data
            obj.X = x; 
        end
        function obj = set_max_iter(obj, i) %--> Set max. iteration number
            obj.max_iter = i;
        end
        function obj = set_train_cost(obj) %--> Set training cost
            obj.J_train = obj.J;
        end
        function obj = set_cv_cost(obj) %--> Set cross-validation cost
            obj.J_cv = obj.J;
        end
        function obj = set_cv_acc(obj, a) %--> Set cross-validation accuracy
            obj.Acc_cv = a;
        end
        function obj = set_thetas(obj, Th) %--> Set the weights of all layers
            len_th = 0; %--> Length of weights vector spanned so far
            for i = 1 : size(obj.Layers, 2) - 1
                th_size = size(get_theta(obj.Layers(i)), 1) * ...
                    size(get_theta(obj.Layers(i)), 2); 
             %--> Portion of weights vector length related to current layer
                th_i = reshape(Th(len_th + 1 : len_th + th_size), ...
                    size(get_theta(obj.Layers(i)), 1), ...
                    size(get_theta(obj.Layers(i)), 2)); 
                                                   %--> Reshape into matrix
                obj.Layers(i) = set_theta(obj.Layers(i), th_i); 
                                                 %--> Set new layer weights
                len_th = len_th + th_size; 
                                     %--> Update lengths of spanned weights
            end
        end
        function obj = cost(obj, y) %--> Update cost
            % Setup some useful variables
            m = size(obj.X, 1); %--> Number of training examples
            
            % Feed-forward to get activations
            In = [ones(m, 1), obj.X]; %--> Input features + Bias
            obj.Layers(1) = set_a(obj.Layers(1), ...
                In); %--> Set activations for 1st layer
            for i = 2 : size(obj.Layers, 2) - 1
                a_i = zeros(size(get_a(obj.Layers(i - 1)), 1), ...
                    size(get_theta(obj.Layers(i - 1)), 1) + 1);
                a_i(:, 2 : end) = sigmoid(get_a(obj.Layers(i - 1)) ...
                    * get_theta(obj.Layers(i - 1)).'); 
                                       %--> Layer activation output portion
                a_i(:, 1) = ones(size(a_i, 1), 1); %--> Bias portion
                obj.Layers(i) = set_a(obj.Layers(i), a_i); 
                                         %--> Set activations for ith layer
            end
            H = sigmoid(get_a(obj.Layers(end - 1)) ...
                    * get_theta(obj.Layers(end - 1)).'); 
                                  %--> Output layer activation "Hypothesis"
            obj.Layers(end) = set_a(obj.Layers(end), H); 
                                      %--> Set activations for output layer

            % Cost
            Y = zeros(m, obj.num_labels); %--> Initialize output matrix
            for i = 1 : obj.num_labels
                Y(:, i) = y == i; %--> Multi-class NN experimental output
            end
            obj.J = (-1 / m) * ...
                sum(sum(Y .* log(H) + (1 - Y) .* log(1 - H))); 
                                                  %--> Unregularized cost
            for i = 1 : size(obj.Layers, 2)
                theta = get_theta(obj.Layers(i));
                obj.J = obj.J + (obj.lambda / (2 * m)) * ...
                    (sum(sum(theta(:, 2 : end).^2))); %--> Regularization
            end
            
            % Backpropagation 
            % Get partial derivatives
            obj.Layers(end) = set_delta(obj.Layers(end), ...
                get_a(obj.Layers(end)) - Y); %--> Last derivative
            for i = size(obj.Layers, 2) - 1 : -1 : 2
                delta_i = get_delta(obj.Layers(i + 1)) * ...
                    get_theta(obj.Layers(i)); 
                delta_i = delta_i(:, 2 : end) .* ...
                    sigmoidGradient(get_a(obj.Layers(i - 1)) ...
                    * get_theta(obj.Layers(i - 1)).'); 
                                       %--> Hidden layer partial derivative
                obj.Layers(i) = set_delta(obj.Layers(i), delta_i);
            end

            % Get gradients
            len_grad = 0; %--> The length of gradiant filled so far 
            for i = 1 : size(obj.Layers, 2) - 1
                % Compute layer gradient
                Delta_i = get_delta(obj.Layers(i + 1)).' * ...
                    get_a(obj.Layers(i));
                Thetai_grad = Delta_i / m;
                Thetai = get_theta(obj.Layers(i));
                Thetai_grad(:, 2 : end) = Thetai_grad(:, 2 : end) + ...
                    (obj.lambda / m) * Thetai(:, 2 : end);
                obj.Layers(i) = set_grad(obj.Layers(i), Thetai_grad); 
                                                    %--> Set layer gradient

                % Fill network gradient
                obj.Grad(len_grad + 1 : len_grad + ...
                    size(get_grad(obj.Layers(i)), 1) * ...
                    size(get_grad(obj.Layers(i)), 2), 1) = Thetai_grad(:);
                len_grad = len_grad + ...
                    size(get_grad(obj.Layers(i)), 1) * ...
                    size(get_grad(obj.Layers(i)), 2);
            end
        end
        function obj = Train(obj, y)
            % Get network weights in a vector form
            Thetas = [];
            for i = 1 : size(obj.Layers, 2) - 1
                thetai = get_theta(obj.Layers(i));
                Thetas = [Thetas; thetai(:)];
            end

            % Minimization algorithm
            RHO = 0.01; %--> Bunch of constants for line searches
            SIG = 0.5; %--> Constants in the Wolfe-Powell conditions
            INT = 0.1; 
       %--> Don't reevaluate within 0.1 of the limit of the current bracket
            EXT = 3.0; %--> Extrapolate maximum 3 times the current bracket
            MAX = 20; %--> Max 20 function evaluations per line search
            RATIO = 100; %--> Maximum allowed slope ratio
            red = 1;
            S = ['Iteration '];
            i = 0; %--> Zero the run length counter
            ls_failed = 0; %--> No previous line search has failed
            fX = [];
            obj = cost(obj, y); %--> Update cost
            f1 = obj.J; df1 = obj.Grad; %--> Get cost and gradient
            i = i + (obj.max_iter < 0); %--> Count epochs?!
            s = -df1; %--> Search direction is steepest
            d1 = -s' * s; %--> This is the slope
            z1 = red / (1 - d1); %--> Initial step is red/(|s|+1)

            while i < abs(obj.max_iter) %--> While not finished
                i = i + (obj.max_iter > 0); %--> Count iterations?!            
                X0 = Thetas; f0 = f1; df0 = df1; 
                                         %--> Make a copy of current values
                Thetas = Thetas + z1 * s; %--> Begin line search
                obj = set_thetas(obj, Thetas); 
                                         %--> Set the weights of all layers
                obj = cost(obj, y); %--> Update cost
                f2 = obj.J; df2 = obj.Grad; %--> Get cost and gradient
                i = i + (obj.max_iter < 0); %--> Count epochs?!
                d2 = df2' * s;
                f3 = f1; d3 = d1; z3 = -z1; 
                                   %--> initialize point 3 equal to point 1
                if obj.max_iter > 0, M = MAX; else M = min(MAX, ...
                        -obj.max_iter - i); end
                success = 0; limit = -1; %--> Initialize quantities
                while 1
                  while ((f2 > f1 + z1 * RHO * d1) || ...
                          (d2 > -SIG * d1)) && (M > 0) 
                    limit = z1; %--> Tighten the bracket
                    if f2 > f1
                      z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3); 
                                                         %--> Quadratic fit
                    else
                      A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3); %-> Cubic fit
                      B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
                      z2 = (sqrt(B * B - A * d2 * z3 * z3) - B) / A;       
                                        %--> Numerical error possible - ok!
                    end
                    if isnan(z2) || isinf(z2)
                      z2 = z3 / 2; 
                             %--> If we had a numerical problem then bisect
                    end
                    z2 = max(min(z2, INT * z3),(1 - INT) * z3);  
                                      %--> Don't accept too close to limits
                    z1 = z1 + z2; %--> Update the step
                    Thetas = Thetas + z2 * s;
                    obj = set_thetas(obj, Thetas); %--> Set the weights of all layers
                    obj = cost(obj, y); %--> Update cost
                    f2 = obj.J; df2 = obj.Grad; %--> Get cost and gradient
                    M = M - 1; i = i + (obj.max_iter < 0); 
                                                        %--> Count epochs?!
                    d2 = df2' * s;
                    z3 = z3 - z2; 
                              %--> z3 is now relative to the location of z2
                  end
                  if f2 > f1 + z1 * RHO * d1 || d2 > -SIG * d1
                    break; %--> This is a failure
                  elseif d2 > SIG * d1
                    success = 1; break; %--> Success
                  elseif M == 0
                    break; %--> Failure
                  end
                  A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3); 
                                              %--> Make cubic extrapolation
                  B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
                  z2 = -d2 * z3 * z3 / ...
                      (B + sqrt(B * B - A * d2 * z3 * z3));        
                                             %--> Num. error possible - ok!
                  if ~isreal(z2) || isnan(z2) || isinf(z2) || z2 < 0 
                                               %--> Num prob or wrong sign?
                    if limit < -0.5 %--> If we have no upper limit
                      z2 = z1 * (EXT-1); 
                                    %--> The extrapolate the maximum amount
                    else
                      z2 = (limit - z1) / 2; %--> Otherwise bisect
                    end
                  elseif (limit > -0.5) && (z2+z1 > limit) 
                                              %--> Extraplation beyond max?
                    z2 = (limit-z1)/2; %--> Bisect
                  elseif (limit < -0.5) && (z2 + z1 > z1 * EXT)       
                                            %--> Extrapolation beyond limit
                    z2 = z1 * (EXT - 1.0); %--> Set to extrapolation limit
                  elseif z2 < -z3 * INT
                    z2 = -z3 * INT;
                  elseif (limit > -0.5) && ...
                          (z2 < (limit - z1) * (1.0 - INT))
                                                   %--> Too close to limit?
                    z2 = (limit - z1) * (1.0 - INT);
                  end
                  f3 = f2; d3 = d2; z3 = -z2; 
                                          %--> Set point 3 equal to point 2
                  z1 = z1 + z2; Thetas = Thetas + z2 * s; 
                                              %--> Update current estimates
                  obj = set_thetas(obj, Thetas); 
                                         %--> Set the weights of all layers
                  obj = cost(obj, y); %--> Update cost
                  f2 = obj.J; df2 = obj.Grad; %--> Get cost and gradient
                  M = M - 1; i = i + (obj.max_iter < 0); %--> Count epochs?!
                  d2 = df2' * s;
                end %--> End of line search
            
                if success %--> If line search succeeded
                  f1 = f2; fX = [fX' f1]';
                  s = (df2' * df2 - df1' * df2) / (df1' * df1) * s - df2;      
                                              %--> Polack-Ribiere direction
                  tmp = df1; df1 = df2; df2 = tmp; %--> Swap derivatives
                  d2 = df1' * s;
                  if d2 > 0 %--> New slope must be negative
                    s = -df1; %--> Otherwise use steepest direction
                    d2 = -s' * s;    
                  end
                  z1 = z1 * min(RATIO, d1 / (d2 - realmin));          
                                             %--> Slope ratio but max RATIO
                  d1 = d2;
                  ls_failed = 0; %--> This line search did not fail
                else
                  Thetas = X0; f1 = f0; df1 = df0;  
                          %--> Restore point from before failed line search
                  if ls_failed || i > abs(obj.max_iter)          
                                     %--> Line search failed twice in a row
                    break; %--> Or we ran out of time, so we give up
                  end
                  tmp = df1; df1 = df2; df2 = tmp; %--> Swap derivatives
                  s = -df1; %--> Try steepest
                  d1 = -s' * s;
                  z1 = 1 / (1 - d1);                     
                  ls_failed = 1; %--> This line search failed
                end
                if exist('OCTAVE_VERSION')
                  fflush(stdout);
                end
             end
              fprintf('\n');
        end
    end
end

% Generic functions
function g = sigmoid(z) %--> Compute sigmoid
    g = 1.0 ./ (1.0 + exp(-z));
end
function g = sigmoidGradient(z) %--> Compute sigmoid gradient
    g = sigmoid(z) .* (1 - sigmoid(z));
end

