function y = my_norm(x)

minx = min(x(:));
maxx = max(x(:));
ranx = maxx-minx;

y = (x-minx)/ranx;

