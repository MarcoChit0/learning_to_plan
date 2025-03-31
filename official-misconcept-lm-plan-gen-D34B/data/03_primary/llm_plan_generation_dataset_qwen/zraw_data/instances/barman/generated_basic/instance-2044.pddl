(define (problem prob)
 (:domain barman)
 (:objects 
      shaker449 - shaker
      left right - hand
      shot304 - shot
      ingredient306 ingredient319 ingredient478 ingredient150 - ingredient
      cocktail452 - cocktail
      dispenser339 dispenser49 dispenser214 dispenser122 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker449)
  (ontable shot304)
  (dispenses dispenser339 ingredient306)
  (dispenses dispenser49 ingredient319)
  (dispenses dispenser214 ingredient478)
  (dispenses dispenser122 ingredient150)
  (clean shaker449)
  (clean shot304)
  (empty shaker449)
  (empty shot304)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker449 l0)
  (shaker-level shaker449 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail452 ingredient306)
  (cocktail-part2 cocktail452 ingredient150)
)
 (:goal
  (and
      (contains shot304 cocktail452)
)))
