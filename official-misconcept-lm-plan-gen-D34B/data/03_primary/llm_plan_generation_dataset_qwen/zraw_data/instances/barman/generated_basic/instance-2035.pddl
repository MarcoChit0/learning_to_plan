(define (problem prob)
 (:domain barman)
 (:objects 
      shaker451 - shaker
      left right - hand
      shot442 shot388 - shot
      ingredient3 ingredient72 - ingredient
      cocktail36 - cocktail
      dispenser131 dispenser357 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker451)
  (ontable shot442)
  (ontable shot388)
  (dispenses dispenser131 ingredient3)
  (dispenses dispenser357 ingredient72)
  (clean shaker451)
  (clean shot442)
  (clean shot388)
  (empty shaker451)
  (empty shot442)
  (empty shot388)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker451 l0)
  (shaker-level shaker451 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail36 ingredient3)
  (cocktail-part2 cocktail36 ingredient72)
)
 (:goal
  (and
      (contains shot442 cocktail36)
)))
