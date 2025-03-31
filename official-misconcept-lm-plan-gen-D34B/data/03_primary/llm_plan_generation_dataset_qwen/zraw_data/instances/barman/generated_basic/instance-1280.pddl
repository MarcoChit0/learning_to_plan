(define (problem prob)
 (:domain barman)
 (:objects 
      shaker398 - shaker
      left right - hand
      shot123 - shot
      ingredient98 ingredient66 ingredient427 ingredient430 - ingredient
      cocktail494 - cocktail
      dispenser234 dispenser330 dispenser144 dispenser66 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker398)
  (ontable shot123)
  (dispenses dispenser234 ingredient98)
  (dispenses dispenser330 ingredient66)
  (dispenses dispenser144 ingredient427)
  (dispenses dispenser66 ingredient430)
  (clean shaker398)
  (clean shot123)
  (empty shaker398)
  (empty shot123)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker398 l0)
  (shaker-level shaker398 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail494 ingredient98)
  (cocktail-part2 cocktail494 ingredient430)
)
 (:goal
  (and
      (contains shot123 cocktail494)
)))
