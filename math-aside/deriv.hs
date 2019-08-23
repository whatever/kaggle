nderiv f  h = \x -> (f(x+h)-f(x-h))/2.0/h


f x = 3*x*x
g x = (nderiv f 0.001) x

main = print (g 1)
