(define (problem prob)
 (:domain barman)
 (:objects 
      shaker366 - shaker
      left right - hand
      shot206 shot123 - shot
      ingredient306 ingredient238 - ingredient
      cocktail129 - cocktail
      dispenser89 dispenser381 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker366)
  (ontable shot206)
  (ontable shot123)
  (dispenses dispenser89 ingredient306)
  (dispenses dispenser381 ingredient238)
  (clean shaker366)
  (clean shot206)
  (clean shot123)
  (empty shaker366)
  (empty shot206)
  (empty shot123)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker366 l0)
  (shaker-level shaker366 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail129 ingredient238)
  (cocktail-part2 cocktail129 ingredient306)
)
 (:goal
  (and
      (contains shot206 cocktail129)
)))
