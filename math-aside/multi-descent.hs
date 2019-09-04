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


-- Return the function that computes the directional derivative for f
-- NOTE: There's nothing forcing u to be normal
-- NOTE: I don't know if the / 2.0 is the correct constant here
directionalDeriv f u =
  \x ->
    let
      xr = (add x (scale u h))
      xl = (add x (scale u (-h)))
    in ((f xr) - (f xl)) / h / 2.0
  where h = 0.01


h = directionalDeriv f [1, 0]
g = directionalDeriv f [0, 1]


-- dir
dir i n = [if j == i then 1 else 0 | j <- [0..(n-1)]]

-- Return the basis vectors for a normal hyper plane
basis n = [(dir i n) | i <- [0..(n-1)]]


-- Return a function that numerically approximates the gradients of a function
grad f n =
  \x -> [(g x) | g <- gradients]
  where
    gradients = [(directionalDeriv f u) | u <- (basis n)]


descend f x0 n = x1
  where g = (grad f n)
        h = 0.001
        x1 = (add x0 (scale (g x0) h))



norm xs = sqrt (sum [x^2 | x <- xs])

-- Probably return a local minimum for a vector value function of n-variables
-- NOTE: This can be considerably optimized
gradDescent f x0 n = val
  where
    f' = (grad f n)
    a  = 0.00005
    x1 = (add x0 (scale (f' x0) (-a)))
    val = if norm (f' x1) < 0.0001
             then x1
             else (gradDescent f x1 n)


-- some multivariate function
f2 [x, y, z] = (x-3)^2 + y^2 + z^2 + 8


-- 
f2' = grad f2 3


-- lfg
main = do
  print "Multivariate Gradient Descent"
  -- print (f [1, 2])
  -- print (f' [1, 2, -3])
  print (gradDescent f2 [3, 10, -1] 3)
