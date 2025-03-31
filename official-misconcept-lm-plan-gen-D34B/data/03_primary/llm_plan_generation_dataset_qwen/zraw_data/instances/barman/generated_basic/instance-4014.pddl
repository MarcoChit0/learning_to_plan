(define (problem prob)
 (:domain barman)
 (:objects 
      shaker491 - shaker
      left right - hand
      shot71 shot232 - shot
      ingredient65 ingredient12 ingredient345 - ingredient
      cocktail353 - cocktail
      dispenser210 dispenser270 dispenser35 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker491)
  (ontable shot71)
  (ontable shot232)
  (dispenses dispenser210 ingredient65)
  (dispenses dispenser270 ingredient12)
  (dispenses dispenser35 ingredient345)
  (clean shaker491)
  (clean shot71)
  (clean shot232)
  (empty shaker491)
  (empty shot71)
  (empty shot232)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker491 l0)
  (shaker-level shaker491 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail353 ingredient345)
  (cocktail-part2 cocktail353 ingredient65)
)
 (:goal
  (and
      (contains shot71 cocktail353)
)))
