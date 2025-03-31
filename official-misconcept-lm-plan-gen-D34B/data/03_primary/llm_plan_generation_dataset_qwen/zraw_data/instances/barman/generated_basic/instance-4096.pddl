(define (problem prob)
 (:domain barman)
 (:objects 
      shaker382 - shaker
      left right - hand
      shot471 shot106 - shot
      ingredient72 ingredient401 - ingredient
      cocktail392 - cocktail
      dispenser162 dispenser352 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker382)
  (ontable shot471)
  (ontable shot106)
  (dispenses dispenser162 ingredient72)
  (dispenses dispenser352 ingredient401)
  (clean shaker382)
  (clean shot471)
  (clean shot106)
  (empty shaker382)
  (empty shot471)
  (empty shot106)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker382 l0)
  (shaker-level shaker382 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail392 ingredient72)
  (cocktail-part2 cocktail392 ingredient401)
)
 (:goal
  (and
      (contains shot471 cocktail392)
)))
