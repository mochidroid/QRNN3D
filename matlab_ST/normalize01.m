function [result] = normalize01(x)
x_max = max_all(x);
x_min = min_all(x);

result = (x - x_min) / (x_max - x_min);
end