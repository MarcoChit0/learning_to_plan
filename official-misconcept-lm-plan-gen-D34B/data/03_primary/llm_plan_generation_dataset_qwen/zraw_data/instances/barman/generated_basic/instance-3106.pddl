(define (problem prob)
 (:domain barman)
 (:objects 
      shaker337 - shaker
      left right - hand
      shot114 shot335 - shot
      ingredient197 ingredient262 - ingredient
      cocktail363 - cocktail
      dispenser261 dispenser423 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker337)
  (ontable shot114)
  (ontable shot335)
  (dispenses dispenser261 ingredient197)
  (dispenses dispenser423 ingredient262)
  (clean shaker337)
  (clean shot114)
  (clean shot335)
  (empty shaker337)
  (empty shot114)
  (empty shot335)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker337 l0)
  (shaker-level shaker337 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail363 ingredient262)
  (cocktail-part2 cocktail363 ingredient197)
)
 (:goal
  (and
      (contains shot114 cocktail363)
)))
