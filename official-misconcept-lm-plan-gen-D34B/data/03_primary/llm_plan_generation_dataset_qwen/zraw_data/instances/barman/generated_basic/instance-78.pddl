(define (problem prob)
 (:domain barman)
 (:objects 
      shaker202 - shaker
      left right - hand
      shot59 shot99 - shot
      ingredient46 ingredient434 ingredient43 - ingredient
      cocktail1 - cocktail
      dispenser221 dispenser19 dispenser297 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker202)
  (ontable shot59)
  (ontable shot99)
  (dispenses dispenser221 ingredient46)
  (dispenses dispenser19 ingredient434)
  (dispenses dispenser297 ingredient43)
  (clean shaker202)
  (clean shot59)
  (clean shot99)
  (empty shaker202)
  (empty shot59)
  (empty shot99)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker202 l0)
  (shaker-level shaker202 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient434)
  (cocktail-part2 cocktail1 ingredient43)
)
 (:goal
  (and
      (contains shot59 cocktail1)
)))
