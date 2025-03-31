(define (problem prob)
 (:domain barman)
 (:objects 
      shaker325 - shaker
      left right - hand
      shot394 - shot
      ingredient130 ingredient400 ingredient425 - ingredient
      cocktail384 - cocktail
      dispenser476 dispenser297 dispenser323 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker325)
  (ontable shot394)
  (dispenses dispenser476 ingredient130)
  (dispenses dispenser297 ingredient400)
  (dispenses dispenser323 ingredient425)
  (clean shaker325)
  (clean shot394)
  (empty shaker325)
  (empty shot394)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker325 l0)
  (shaker-level shaker325 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail384 ingredient425)
  (cocktail-part2 cocktail384 ingredient130)
)
 (:goal
  (and
      (contains shot394 cocktail384)
)))
