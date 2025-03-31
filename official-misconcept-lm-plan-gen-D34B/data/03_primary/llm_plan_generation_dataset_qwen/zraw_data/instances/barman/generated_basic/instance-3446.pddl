(define (problem prob)
 (:domain barman)
 (:objects 
      shaker103 - shaker
      left right - hand
      shot282 shot128 - shot
      ingredient57 ingredient432 ingredient235 ingredient183 - ingredient
      cocktail83 - cocktail
      dispenser264 dispenser368 dispenser208 dispenser119 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker103)
  (ontable shot282)
  (ontable shot128)
  (dispenses dispenser264 ingredient57)
  (dispenses dispenser368 ingredient432)
  (dispenses dispenser208 ingredient235)
  (dispenses dispenser119 ingredient183)
  (clean shaker103)
  (clean shot282)
  (clean shot128)
  (empty shaker103)
  (empty shot282)
  (empty shot128)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker103 l0)
  (shaker-level shaker103 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail83 ingredient57)
  (cocktail-part2 cocktail83 ingredient432)
)
 (:goal
  (and
      (contains shot282 cocktail83)
)))
