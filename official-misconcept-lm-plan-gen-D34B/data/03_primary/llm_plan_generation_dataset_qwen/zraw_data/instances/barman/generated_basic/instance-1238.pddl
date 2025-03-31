(define (problem prob)
 (:domain barman)
 (:objects 
      shaker491 - shaker
      left right - hand
      shot128 shot457 shot442 - shot
      ingredient473 ingredient94 ingredient354 ingredient182 - ingredient
      cocktail450 - cocktail
      dispenser337 dispenser226 dispenser5 dispenser393 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker491)
  (ontable shot128)
  (ontable shot457)
  (ontable shot442)
  (dispenses dispenser337 ingredient473)
  (dispenses dispenser226 ingredient94)
  (dispenses dispenser5 ingredient354)
  (dispenses dispenser393 ingredient182)
  (clean shaker491)
  (clean shot128)
  (clean shot457)
  (clean shot442)
  (empty shaker491)
  (empty shot128)
  (empty shot457)
  (empty shot442)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker491 l0)
  (shaker-level shaker491 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail450 ingredient473)
  (cocktail-part2 cocktail450 ingredient182)
)
 (:goal
  (and
      (contains shot128 cocktail450)
      (contains shot457 ingredient182)
)))
