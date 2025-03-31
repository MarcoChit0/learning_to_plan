(define (problem prob)
 (:domain barman)
 (:objects 
      shaker13 - shaker
      left right - hand
      shot46 shot405 - shot
      ingredient349 ingredient70 ingredient66 ingredient236 - ingredient
      cocktail1 - cocktail
      dispenser448 dispenser484 dispenser427 dispenser224 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker13)
  (ontable shot46)
  (ontable shot405)
  (dispenses dispenser448 ingredient349)
  (dispenses dispenser484 ingredient70)
  (dispenses dispenser427 ingredient66)
  (dispenses dispenser224 ingredient236)
  (clean shaker13)
  (clean shot46)
  (clean shot405)
  (empty shaker13)
  (empty shot46)
  (empty shot405)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker13 l0)
  (shaker-level shaker13 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient66)
  (cocktail-part2 cocktail1 ingredient349)
)
 (:goal
  (and
      (contains shot46 cocktail1)
)))
