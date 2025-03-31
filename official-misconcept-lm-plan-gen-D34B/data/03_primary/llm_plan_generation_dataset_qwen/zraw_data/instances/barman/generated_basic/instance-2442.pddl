(define (problem prob)
 (:domain barman)
 (:objects 
      shaker108 - shaker
      left right - hand
      shot1 shot270 shot408 - shot
      ingredient35 ingredient453 ingredient446 - ingredient
      cocktail218 - cocktail
      dispenser304 dispenser476 dispenser454 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker108)
  (ontable shot1)
  (ontable shot270)
  (ontable shot408)
  (dispenses dispenser304 ingredient35)
  (dispenses dispenser476 ingredient453)
  (dispenses dispenser454 ingredient446)
  (clean shaker108)
  (clean shot1)
  (clean shot270)
  (clean shot408)
  (empty shaker108)
  (empty shot1)
  (empty shot270)
  (empty shot408)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker108 l0)
  (shaker-level shaker108 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail218 ingredient35)
  (cocktail-part2 cocktail218 ingredient453)
)
 (:goal
  (and
      (contains shot1 cocktail218)
      (contains shot270 ingredient446)
)))
