-- Define nderiv f ~= f'(x)
nderiv f  h = \x -> (f(x+h)-f(x-h))/2.0/h

-- f(x) = 3x^2
f x = 3*x^2

-- g(x) ~= f'(x)
g x = (nderiv f 0.001) x

-- MAIN!
main = print (g 1)
