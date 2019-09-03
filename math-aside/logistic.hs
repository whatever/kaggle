-- Logistic regression implemented in Haskell
-- for fun... I guess


-- Return logistic(x)
logistic x = 1.0 / (1.0 - (exp (-x)))



-- Return dot product between x and y
-- NOTE: I don't really k now how sum is implemented here
dot xs ys = sum [x*y | (x, y) <- (zip xs ys)]


-- Return linear function for theta
linearFunction theta = (\x -> (dot theta x))


-- Affine linear function
affineLinearFunction theta = (\x -> (head theta) + (dot (tail theta) x))


-- f(x) = 3.0 x1 + 1.2 x2 + 10 x3
f = (linearFunction [3.0, 1.2, 10.0])


-- g(x) = 2.0 + 2.0 x1
g = (affineLinearFunction [2.0, 2.0, 2.0])


-- Metric used to compute whether f(x) predicts y
-- Make this more efficient? Will this sum things up iteratively?
-- l2-norm
cost f xs ys = (sqrt (sum [((f x)-y)^2 | (x, y) <- (zip xs ys)])) / (fromIntegral (length xs)) / 2.0


-- X: input vector
xs = [[1, 2] | x <- [1..4]]


-- Y: output vector
ys = [4 | y <- [1..4]]


-- Basic logistic regression
-- =========================
-- 0. Have a matrix X and vector Y.
--    Need to predict what parameters of the logistic function predict f(X) = Y
-- 1. Randomize parameters theta
-- 2. Compute COST of f(X)-Y
-- 3. Compute GRADIENT of cost function (???)
-- 4. Tweak parameters according to gradient descent

-- iterate one step of gradient descent
descend a b c = a

-- train a logistic model for f(xs) = ys
train xs ys =
  if (cost g xs ys) < 0.001
     then g
     else (descend g xs ys)
  




main = do
  -- print [g x | x <- xs]
  -- print [y | y <- ys]
  -- print (sqrt (sum [((g x)-y)^2 | (x, y) <- (zip xs ys)]))
  print (cost g xs ys)
