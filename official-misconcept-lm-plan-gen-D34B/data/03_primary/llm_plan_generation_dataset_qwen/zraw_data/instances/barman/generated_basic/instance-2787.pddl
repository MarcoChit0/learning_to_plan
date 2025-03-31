(define (problem prob)
 (:domain barman)
 (:objects 
      shaker409 - shaker
      left right - hand
      shot253 - shot
      ingredient361 ingredient66 ingredient348 ingredient153 - ingredient
      cocktail32 - cocktail
      dispenser180 dispenser259 dispenser491 dispenser327 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker409)
  (ontable shot253)
  (dispenses dispenser180 ingredient361)
  (dispenses dispenser259 ingredient66)
  (dispenses dispenser491 ingredient348)
  (dispenses dispenser327 ingredient153)
  (clean shaker409)
  (clean shot253)
  (empty shaker409)
  (empty shot253)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker409 l0)
  (shaker-level shaker409 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail32 ingredient66)
  (cocktail-part2 cocktail32 ingredient361)
)
 (:goal
  (and
      (contains shot253 cocktail32)
)))
