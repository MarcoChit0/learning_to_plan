(define (problem prob)
 (:domain barman)
 (:objects 
      shaker85 - shaker
      left right - hand
      shot408 shot120 - shot
      ingredient6 ingredient362 ingredient112 ingredient475 - ingredient
      cocktail1 - cocktail
      dispenser218 dispenser373 dispenser243 dispenser314 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker85)
  (ontable shot408)
  (ontable shot120)
  (dispenses dispenser218 ingredient6)
  (dispenses dispenser373 ingredient362)
  (dispenses dispenser243 ingredient112)
  (dispenses dispenser314 ingredient475)
  (clean shaker85)
  (clean shot408)
  (clean shot120)
  (empty shaker85)
  (empty shot408)
  (empty shot120)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker85 l0)
  (shaker-level shaker85 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient475)
  (cocktail-part2 cocktail1 ingredient6)
)
 (:goal
  (and
      (contains shot408 cocktail1)
)))
