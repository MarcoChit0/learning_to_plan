(define (problem prob)
 (:domain barman)
 (:objects 
      shaker159 - shaker
      left right - hand
      shot328 - shot
      ingredient317 ingredient138 - ingredient
      cocktail240 - cocktail
      dispenser477 dispenser120 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker159)
  (ontable shot328)
  (dispenses dispenser477 ingredient317)
  (dispenses dispenser120 ingredient138)
  (clean shaker159)
  (clean shot328)
  (empty shaker159)
  (empty shot328)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker159 l0)
  (shaker-level shaker159 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail240 ingredient138)
  (cocktail-part2 cocktail240 ingredient317)
)
 (:goal
  (and
      (contains shot328 cocktail240)
)))
