(define (problem prob)
 (:domain barman)
 (:objects 
      shaker452 - shaker
      left right - hand
      shot381 shot273 shot425 - shot
      ingredient495 ingredient101 ingredient236 - ingredient
      cocktail457 - cocktail
      dispenser4 dispenser237 dispenser86 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker452)
  (ontable shot381)
  (ontable shot273)
  (ontable shot425)
  (dispenses dispenser4 ingredient495)
  (dispenses dispenser237 ingredient101)
  (dispenses dispenser86 ingredient236)
  (clean shaker452)
  (clean shot381)
  (clean shot273)
  (clean shot425)
  (empty shaker452)
  (empty shot381)
  (empty shot273)
  (empty shot425)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker452 l0)
  (shaker-level shaker452 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail457 ingredient495)
  (cocktail-part2 cocktail457 ingredient101)
)
 (:goal
  (and
      (contains shot381 cocktail457)
      (contains shot273 cocktail457)
)))
