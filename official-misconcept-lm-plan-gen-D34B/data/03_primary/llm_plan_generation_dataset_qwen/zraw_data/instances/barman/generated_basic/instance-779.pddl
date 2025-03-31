(define (problem prob)
 (:domain barman)
 (:objects 
      shaker99 - shaker
      left right - hand
      shot3 - shot
      ingredient415 ingredient238 - ingredient
      cocktail1 - cocktail
      dispenser224 dispenser428 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker99)
  (ontable shot3)
  (dispenses dispenser224 ingredient415)
  (dispenses dispenser428 ingredient238)
  (clean shaker99)
  (clean shot3)
  (empty shaker99)
  (empty shot3)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker99 l0)
  (shaker-level shaker99 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient415)
  (cocktail-part2 cocktail1 ingredient238)
)
 (:goal
  (and
      (contains shot3 cocktail1)
)))
