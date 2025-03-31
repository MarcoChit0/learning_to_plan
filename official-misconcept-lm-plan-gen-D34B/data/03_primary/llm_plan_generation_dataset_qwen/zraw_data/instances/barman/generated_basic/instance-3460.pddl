(define (problem prob)
 (:domain barman)
 (:objects 
      shaker180 - shaker
      left right - hand
      shot129 shot357 shot79 - shot
      ingredient211 ingredient391 ingredient282 ingredient69 - ingredient
      cocktail450 - cocktail
      dispenser440 dispenser428 dispenser314 dispenser83 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker180)
  (ontable shot129)
  (ontable shot357)
  (ontable shot79)
  (dispenses dispenser440 ingredient211)
  (dispenses dispenser428 ingredient391)
  (dispenses dispenser314 ingredient282)
  (dispenses dispenser83 ingredient69)
  (clean shaker180)
  (clean shot129)
  (clean shot357)
  (clean shot79)
  (empty shaker180)
  (empty shot129)
  (empty shot357)
  (empty shot79)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker180 l0)
  (shaker-level shaker180 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail450 ingredient69)
  (cocktail-part2 cocktail450 ingredient391)
)
 (:goal
  (and
      (contains shot129 cocktail450)
      (contains shot357 cocktail450)
)))
