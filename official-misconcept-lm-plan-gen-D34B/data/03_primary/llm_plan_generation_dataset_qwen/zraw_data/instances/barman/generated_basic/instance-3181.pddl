(define (problem prob)
 (:domain barman)
 (:objects 
      shaker261 - shaker
      left right - hand
      shot397 - shot
      ingredient310 ingredient342 ingredient340 - ingredient
      cocktail489 - cocktail
      dispenser66 dispenser193 dispenser172 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker261)
  (ontable shot397)
  (dispenses dispenser66 ingredient310)
  (dispenses dispenser193 ingredient342)
  (dispenses dispenser172 ingredient340)
  (clean shaker261)
  (clean shot397)
  (empty shaker261)
  (empty shot397)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker261 l0)
  (shaker-level shaker261 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail489 ingredient340)
  (cocktail-part2 cocktail489 ingredient310)
)
 (:goal
  (and
      (contains shot397 cocktail489)
)))
