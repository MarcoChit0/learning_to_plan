(define (problem prob)
 (:domain barman)
 (:objects 
      shaker382 - shaker
      left right - hand
      shot61 shot465 - shot
      ingredient482 ingredient493 ingredient225 ingredient359 - ingredient
      cocktail123 - cocktail
      dispenser208 dispenser198 dispenser295 dispenser432 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker382)
  (ontable shot61)
  (ontable shot465)
  (dispenses dispenser208 ingredient482)
  (dispenses dispenser198 ingredient493)
  (dispenses dispenser295 ingredient225)
  (dispenses dispenser432 ingredient359)
  (clean shaker382)
  (clean shot61)
  (clean shot465)
  (empty shaker382)
  (empty shot61)
  (empty shot465)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker382 l0)
  (shaker-level shaker382 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail123 ingredient482)
  (cocktail-part2 cocktail123 ingredient359)
)
 (:goal
  (and
      (contains shot61 cocktail123)
)))
