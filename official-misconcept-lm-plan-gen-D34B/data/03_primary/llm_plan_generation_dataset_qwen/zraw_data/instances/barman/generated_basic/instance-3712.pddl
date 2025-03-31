(define (problem prob)
 (:domain barman)
 (:objects 
      shaker39 - shaker
      left right - hand
      shot371 - shot
      ingredient37 ingredient218 ingredient454 ingredient271 - ingredient
      cocktail352 - cocktail
      dispenser13 dispenser106 dispenser332 dispenser487 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker39)
  (ontable shot371)
  (dispenses dispenser13 ingredient37)
  (dispenses dispenser106 ingredient218)
  (dispenses dispenser332 ingredient454)
  (dispenses dispenser487 ingredient271)
  (clean shaker39)
  (clean shot371)
  (empty shaker39)
  (empty shot371)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker39 l0)
  (shaker-level shaker39 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail352 ingredient454)
  (cocktail-part2 cocktail352 ingredient218)
)
 (:goal
  (and
      (contains shot371 cocktail352)
)))
