-- Multivariate Gradient Descent
-- -----------------------------
-- Experimenting with writing Haskell for something that I used to know but
-- completely forget.


-- ...
import System.IO (isEOF)
import Control.Monad (forever, when)
import System.Exit (exitSuccess)


-- Add to vectors
add :: [Double] -> [Double] -> [Double]
add xs ys = [x+y | (x, y) <- (zip xs ys)]


-- scale
scale :: [Double] -> Double -> [Double]
scale xs c = [c*x | x <- xs]


-- Return the function that computes the directional derivative for f
-- NOTE: There's nothing forcing u to be normal
-- NOTE: I don't know if the / 2.0 is the correct constant here
directionalDeriv :: ([Double] -> Double) -> [Double] -> [Double] -> Double
directionalDeriv f u =
  \x ->
    let
      xr = (add x (scale u h))
      xl = (add x (scale u (-h)))
    in ((f xr) - (f xl)) / h / 2.0
  where h = 0.01


-- Return the basis vectors for a standard euclidean space
basis :: Int -> [[Double]]
basis n = [(dir i n) | i <- [0..(n-1)]]
  where dir i n = [if j == i then 1 else 0 | j <- [0..(n-1)]]


-- Return a function that numerically approximates the gradients of a function
-- NOTE: This just returns a vector the directional derivatives with respect to the standard basis
grad :: ([Double] -> Double) -> Int -> [Double] -> [Double]
grad f n =
  \x -> [(g x) | g <- gradients]
  where
    gradients = [(directionalDeriv f u) | u <- (basis n)]


-- Return the l2-norm of a vector
norm :: [Double] -> Double
norm xs = sqrt (sum [x^2 | x <- xs])


-- single iteration of gradient descent
-- NOTE: This is un-used
descend :: ([Double] -> Double) -> [Double] -> Int -> [Double]
descend f x0 n = x1
  where g = (grad f n)
        h = 0.001
        x1 = (add x0 (scale (g x0) h))


-- Probably return a local minimum for a vector value function of n-variables
-- NOTE: This can be considerably optimized
gradDescentOld :: ([Double] -> Double) -> [Double] -> Int -> [Double]
gradDescentOld f x0 n = val
  where
    f' = (grad f n)
    a  = 0.00005
    x1 = (add x0 (scale (f' x0) (-a)))
    val = if norm (f' x1) < 0.0001
             then x1
             else (gradDescentOld f x1 n)

-- Return a near-local minimum for a vector-valued real function
-- Probably return a local minimum for a vector value function of n-variables
-- NOTE: This can be considerably optimized
gradDescent :: ([Double] -> Double) -> [Double] -> Int -> [Double]
gradDescent f x0 n =
  (descend x0)
  where f' = (grad f n)
        a  = 0.00005
        h  = 0.0001
        iterate x = (add x (scale (f' x) (-a)))
        descend x =
          if (norm (f' x)) < h
             then x
             else (descend (iterate x))


-- some multivariate function
f2 :: [Double] -> Double
f2 [x, y, z] = (x-3)^2 + (y+2)^2 + (z-9)^2 + 8


-- Load in a csv files as a matrix
loadCsvAsMatrix :: [Char] -> [[Double]]
loadCsvAsMatrix fileName = [[1]]


-- readMatrix will return a matrix of doubles from STDIN
readMatrix content =
  let ls = lines content
  in [[1.0]]



annoyingSeq str = [remainder | remainder <- (dropWhile (/= ' ') str)]


-- Ugly csv line parser that we use just because we wanted to roll our own
split :: String -> [String]
split "" = []
split str = [piece] ++ (split remaining)
  -- TODO: figure out the values for piece/rest simultaneously
  where piece = (takeWhile (/= ',') str)
        rest = (dropWhile (/= ',') str)
        remaining = if null rest then "" else (tail rest)



-- lfg
main = do
  print "Multivariate Gradient Descent"

  -- print (gradDescent f2 [3, 10, -1] 3)
  -- print "Find the local minimum of (x-3)^2 + y^2 + z^2 + 8"
  -- print "Equals ="
  -- print (gradDescent f2 [30000, -1000212, -1687878] 3)


  -- contents <- getContents
  line <- getLine
  print (split line)

  line <- getLine
  print (split line)
