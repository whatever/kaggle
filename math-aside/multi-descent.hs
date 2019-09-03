

-- Add to vectors
add xs ys = [x+y | (x, y) <- (zip xs ys)]


-- scale
scale xs c = [c*x | x <- xs]


-- Retrurn numerical derivative for f
nderiv f = (\x -> ((f (add x xr)) - (f (add x xl)))/2.0/h)
  where 
        h  = 0.001
        xr = [0,  h]
        xl = [0, -h]


-- f(x)
f [x, y] = (x-3)^2 + (y+2)^2


-- grad f(x)
fx = nderiv f


directionalDeriv f u =
  \x ->
    let
      xr = (add x (scale u h))
      xl = (add x (scale u (-h)))
    in ((f xr) - (f xl)) / h / 2.0
  where h = 0.01


h = directionalDeriv f [1, 0]
g = directionalDeriv f [0, 1]


f2 [x, y, z] = 2*x + 3*y^2 + 4*z^3

fg2 = directionalDeriv f2 [0, 0, 1]


-- lfg
main = do
  print "Multivariate Gradient Descent"
  -- print (f [1, 2])
  -- print (f' [1, 2, -3])

  print (fx [3.1, -2.1])
  print (h [2, 0])
  print (g [2, 0])
  print (fg2 [0, 0, 1])
