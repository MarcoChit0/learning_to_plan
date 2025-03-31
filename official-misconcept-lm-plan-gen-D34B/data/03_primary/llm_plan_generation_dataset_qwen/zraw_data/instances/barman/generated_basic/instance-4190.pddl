(define (problem prob)
 (:domain barman)
 (:objects 
      shaker363 - shaker
      left right - hand
      shot457 - shot
      ingredient193 ingredient437 ingredient108 ingredient224 - ingredient
      cocktail119 - cocktail
      dispenser238 dispenser92 dispenser270 dispenser222 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker363)
  (ontable shot457)
  (dispenses dispenser238 ingredient193)
  (dispenses dispenser92 ingredient437)
  (dispenses dispenser270 ingredient108)
  (dispenses dispenser222 ingredient224)
  (clean shaker363)
  (clean shot457)
  (empty shaker363)
  (empty shot457)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker363 l0)
  (shaker-level shaker363 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail119 ingredient437)
  (cocktail-part2 cocktail119 ingredient224)
)
 (:goal
  (and
      (contains shot457 cocktail119)
)))
