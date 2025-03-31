(define (problem prob)
 (:domain barman)
 (:objects 
      shaker347 - shaker
      left right - hand
      shot432 shot495 shot413 - shot
      ingredient71 ingredient373 - ingredient
      cocktail37 - cocktail
      dispenser59 dispenser419 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker347)
  (ontable shot432)
  (ontable shot495)
  (ontable shot413)
  (dispenses dispenser59 ingredient71)
  (dispenses dispenser419 ingredient373)
  (clean shaker347)
  (clean shot432)
  (clean shot495)
  (clean shot413)
  (empty shaker347)
  (empty shot432)
  (empty shot495)
  (empty shot413)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker347 l0)
  (shaker-level shaker347 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail37 ingredient373)
  (cocktail-part2 cocktail37 ingredient71)
)
 (:goal
  (and
      (contains shot432 cocktail37)
      (contains shot495 cocktail37)
)))
