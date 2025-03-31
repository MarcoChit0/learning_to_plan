(define (problem prob)
 (:domain barman)
 (:objects 
      shaker263 - shaker
      left right - hand
      shot246 shot54 - shot
      ingredient151 ingredient418 ingredient29 ingredient121 - ingredient
      cocktail123 - cocktail
      dispenser208 dispenser67 dispenser270 dispenser50 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker263)
  (ontable shot246)
  (ontable shot54)
  (dispenses dispenser208 ingredient151)
  (dispenses dispenser67 ingredient418)
  (dispenses dispenser270 ingredient29)
  (dispenses dispenser50 ingredient121)
  (clean shaker263)
  (clean shot246)
  (clean shot54)
  (empty shaker263)
  (empty shot246)
  (empty shot54)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker263 l0)
  (shaker-level shaker263 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail123 ingredient418)
  (cocktail-part2 cocktail123 ingredient151)
)
 (:goal
  (and
      (contains shot246 cocktail123)
)))
