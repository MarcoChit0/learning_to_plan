(define (problem prob)
 (:domain barman)
 (:objects 
      shaker427 - shaker
      left right - hand
      shot273 shot204 - shot
      ingredient54 ingredient120 - ingredient
      cocktail342 - cocktail
      dispenser365 dispenser133 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker427)
  (ontable shot273)
  (ontable shot204)
  (dispenses dispenser365 ingredient54)
  (dispenses dispenser133 ingredient120)
  (clean shaker427)
  (clean shot273)
  (clean shot204)
  (empty shaker427)
  (empty shot273)
  (empty shot204)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker427 l0)
  (shaker-level shaker427 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail342 ingredient120)
  (cocktail-part2 cocktail342 ingredient54)
)
 (:goal
  (and
      (contains shot273 cocktail342)
)))
