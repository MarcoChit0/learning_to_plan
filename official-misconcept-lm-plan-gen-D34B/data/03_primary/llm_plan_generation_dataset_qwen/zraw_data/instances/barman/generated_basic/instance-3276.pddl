(define (problem prob)
 (:domain barman)
 (:objects 
      shaker51 - shaker
      left right - hand
      shot256 shot75 - shot
      ingredient325 ingredient232 ingredient447 - ingredient
      cocktail484 - cocktail
      dispenser13 dispenser79 dispenser418 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker51)
  (ontable shot256)
  (ontable shot75)
  (dispenses dispenser13 ingredient325)
  (dispenses dispenser79 ingredient232)
  (dispenses dispenser418 ingredient447)
  (clean shaker51)
  (clean shot256)
  (clean shot75)
  (empty shaker51)
  (empty shot256)
  (empty shot75)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker51 l0)
  (shaker-level shaker51 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail484 ingredient447)
  (cocktail-part2 cocktail484 ingredient325)
)
 (:goal
  (and
      (contains shot256 cocktail484)
)))
