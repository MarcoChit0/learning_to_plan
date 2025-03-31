(define (problem prob)
 (:domain barman)
 (:objects 
      shaker364 - shaker
      left right - hand
      shot415 shot146 - shot
      ingredient123 ingredient309 ingredient404 - ingredient
      cocktail310 - cocktail
      dispenser193 dispenser84 dispenser40 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker364)
  (ontable shot415)
  (ontable shot146)
  (dispenses dispenser193 ingredient123)
  (dispenses dispenser84 ingredient309)
  (dispenses dispenser40 ingredient404)
  (clean shaker364)
  (clean shot415)
  (clean shot146)
  (empty shaker364)
  (empty shot415)
  (empty shot146)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker364 l0)
  (shaker-level shaker364 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail310 ingredient404)
  (cocktail-part2 cocktail310 ingredient309)
)
 (:goal
  (and
      (contains shot415 cocktail310)
)))
