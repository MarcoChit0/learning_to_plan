(define (problem prob)
 (:domain barman)
 (:objects 
      shaker48 - shaker
      left right - hand
      shot412 - shot
      ingredient389 ingredient64 ingredient56 ingredient446 - ingredient
      cocktail106 - cocktail
      dispenser58 dispenser486 dispenser67 dispenser454 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker48)
  (ontable shot412)
  (dispenses dispenser58 ingredient389)
  (dispenses dispenser486 ingredient64)
  (dispenses dispenser67 ingredient56)
  (dispenses dispenser454 ingredient446)
  (clean shaker48)
  (clean shot412)
  (empty shaker48)
  (empty shot412)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker48 l0)
  (shaker-level shaker48 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail106 ingredient389)
  (cocktail-part2 cocktail106 ingredient64)
)
 (:goal
  (and
      (contains shot412 cocktail106)
)))
