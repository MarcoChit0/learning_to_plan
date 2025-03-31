(define (problem prob)
 (:domain barman)
 (:objects 
      shaker263 - shaker
      left right - hand
      shot387 shot266 - shot
      ingredient441 ingredient66 ingredient310 - ingredient
      cocktail45 - cocktail
      dispenser153 dispenser45 dispenser131 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker263)
  (ontable shot387)
  (ontable shot266)
  (dispenses dispenser153 ingredient441)
  (dispenses dispenser45 ingredient66)
  (dispenses dispenser131 ingredient310)
  (clean shaker263)
  (clean shot387)
  (clean shot266)
  (empty shaker263)
  (empty shot387)
  (empty shot266)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker263 l0)
  (shaker-level shaker263 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail45 ingredient310)
  (cocktail-part2 cocktail45 ingredient441)
)
 (:goal
  (and
      (contains shot387 cocktail45)
)))
