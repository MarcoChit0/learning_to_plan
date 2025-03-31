(define (problem prob)
 (:domain barman)
 (:objects 
      shaker308 - shaker
      left right - hand
      shot24 - shot
      ingredient179 ingredient457 ingredient288 - ingredient
      cocktail14 - cocktail
      dispenser395 dispenser58 dispenser242 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker308)
  (ontable shot24)
  (dispenses dispenser395 ingredient179)
  (dispenses dispenser58 ingredient457)
  (dispenses dispenser242 ingredient288)
  (clean shaker308)
  (clean shot24)
  (empty shaker308)
  (empty shot24)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker308 l0)
  (shaker-level shaker308 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail14 ingredient457)
  (cocktail-part2 cocktail14 ingredient179)
)
 (:goal
  (and
      (contains shot24 cocktail14)
)))
