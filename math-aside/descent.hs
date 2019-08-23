-- Return n!
-- XXX: Obligatory recursive implementation
factorial n =
  if n < 2
  then 1
  else n*(factorial (n-1))


-- Return an approximation for the derivative of a function
nderiv f  h = \x -> (f(x+h)-f(x-h))/2.0/h


-- Return an iteration
descend f x0 step = (f x0) - (step * ((nderiv f 0.001) x0))


-- Return x0 s.t min f(x) is "minimized"
-- NOTE: This probably isn't smart enough to cache f'
gradDescent f x0 = val
  where
    f' = (nderiv f 0.0001)
    x1 = x0 - 0.01 * (f' x0)
    val = if abs (f' x1) < 0.001
    then
      x1
    else
      gradDescent f x1


-- DEMO FUNCTION
f x = (x-3000.0)^2


-- x_x
main = do
  -- print [factorial x | x <- [1..10]]
  -- print (descend f 4 0.01)
  print (gradDescent f 10)
