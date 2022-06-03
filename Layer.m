classdef Layer
    properties(SetAccess = private)
        Act_fn %--> Activation function type
        Theta %--> Weights on the output side of the layer
        delta %--> Layer partial derivative
        a_l %--> Activation values matrix of layer
        grad %--> Layer gradient
    end
    methods
        function obj = Layer(act_fn, sl, sl_plus) %--> Layer constructor
            % Initialize layer output weights
            if(sl_plus > 0)
                epsilon_init = sqrt(6) / ...
                    sqrt(sl + sl_plus); %--> Weight limit
                obj.Theta = rand(sl_plus, sl + 1) * 2 * epsilon_init ...
                    - epsilon_init; %--> Initialize weights
                obj.grad = zeros(size(obj.Theta)); %--> Initialize gradient
            else
                obj.Theta = []; %--> Output neuron doesn't have weights
                obj.grad = []; %--> No associated gradient
            end

            % Empty initial partial derivative
            obj.delta = [];

            % Initialize activation function
            obj.Act_fn = act_fn; %--> Assign activation function type
        end
        function obj = set_a(obj, act) %--> Set activation values
            obj.a_l = act;
        end
        function obj = set_delta(obj, delt) %--> Set partial derivative
            obj.delta = delt;
        end
        function obj = set_theta(obj, tht) %--> Set layer weights
            obj.Theta = tht;
        end
        function obj = set_grad(obj, g) %--> Set layer gradient
            obj.grad = g;
        end
        function act = get_a(obj) %--> Get activation values
            act = obj.a_l;
        end
        function delt = get_delta(obj) %--> Get partial derivative
            delt = obj.delta;
        end
        function tht = get_theta(obj) %--> Get layer weights
            tht = obj.Theta;
        end
        function g = get_grad(obj) %--> Get layer gradient
            g = obj.grad;
        end
    end
end