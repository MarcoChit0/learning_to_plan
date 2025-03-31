(define (problem prob)
 (:domain barman)
 (:objects 
      shaker373 - shaker
      left right - hand
      shot452 - shot
      ingredient282 ingredient432 - ingredient
      cocktail350 - cocktail
      dispenser333 dispenser106 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker373)
  (ontable shot452)
  (dispenses dispenser333 ingredient282)
  (dispenses dispenser106 ingredient432)
  (clean shaker373)
  (clean shot452)
  (empty shaker373)
  (empty shot452)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker373 l0)
  (shaker-level shaker373 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail350 ingredient282)
  (cocktail-part2 cocktail350 ingredient432)
)
 (:goal
  (and
      (contains shot452 cocktail350)
)))
