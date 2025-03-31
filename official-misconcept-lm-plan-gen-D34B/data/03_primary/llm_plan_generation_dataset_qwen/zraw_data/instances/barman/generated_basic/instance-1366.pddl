(define (problem prob)
 (:domain barman)
 (:objects 
      shaker385 - shaker
      left right - hand
      shot64 - shot
      ingredient14 ingredient190 ingredient202 - ingredient
      cocktail484 - cocktail
      dispenser290 dispenser328 dispenser296 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker385)
  (ontable shot64)
  (dispenses dispenser290 ingredient14)
  (dispenses dispenser328 ingredient190)
  (dispenses dispenser296 ingredient202)
  (clean shaker385)
  (clean shot64)
  (empty shaker385)
  (empty shot64)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker385 l0)
  (shaker-level shaker385 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail484 ingredient202)
  (cocktail-part2 cocktail484 ingredient190)
)
 (:goal
  (and
      (contains shot64 cocktail484)
)))
