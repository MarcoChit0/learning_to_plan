(define (problem prob)
 (:domain barman)
 (:objects 
      shaker37 - shaker
      left right - hand
      shot402 shot108 - shot
      ingredient383 ingredient28 ingredient429 ingredient347 - ingredient
      cocktail251 - cocktail
      dispenser229 dispenser499 dispenser398 dispenser100 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker37)
  (ontable shot402)
  (ontable shot108)
  (dispenses dispenser229 ingredient383)
  (dispenses dispenser499 ingredient28)
  (dispenses dispenser398 ingredient429)
  (dispenses dispenser100 ingredient347)
  (clean shaker37)
  (clean shot402)
  (clean shot108)
  (empty shaker37)
  (empty shot402)
  (empty shot108)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker37 l0)
  (shaker-level shaker37 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail251 ingredient383)
  (cocktail-part2 cocktail251 ingredient28)
)
 (:goal
  (and
      (contains shot402 cocktail251)
)))
