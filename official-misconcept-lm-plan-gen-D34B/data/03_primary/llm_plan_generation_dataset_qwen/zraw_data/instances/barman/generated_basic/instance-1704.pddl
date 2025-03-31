(define (problem prob)
 (:domain barman)
 (:objects 
      shaker311 - shaker
      left right - hand
      shot95 - shot
      ingredient46 ingredient283 ingredient479 ingredient117 - ingredient
      cocktail73 - cocktail
      dispenser198 dispenser351 dispenser225 dispenser397 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker311)
  (ontable shot95)
  (dispenses dispenser198 ingredient46)
  (dispenses dispenser351 ingredient283)
  (dispenses dispenser225 ingredient479)
  (dispenses dispenser397 ingredient117)
  (clean shaker311)
  (clean shot95)
  (empty shaker311)
  (empty shot95)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker311 l0)
  (shaker-level shaker311 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail73 ingredient46)
  (cocktail-part2 cocktail73 ingredient117)
)
 (:goal
  (and
      (contains shot95 cocktail73)
)))
