(define (problem prob)
 (:domain barman)
 (:objects 
      shaker408 - shaker
      left right - hand
      shot90 shot132 shot71 - shot
      ingredient358 ingredient423 ingredient439 - ingredient
      cocktail197 - cocktail
      dispenser366 dispenser238 dispenser207 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker408)
  (ontable shot90)
  (ontable shot132)
  (ontable shot71)
  (dispenses dispenser366 ingredient358)
  (dispenses dispenser238 ingredient423)
  (dispenses dispenser207 ingredient439)
  (clean shaker408)
  (clean shot90)
  (clean shot132)
  (clean shot71)
  (empty shaker408)
  (empty shot90)
  (empty shot132)
  (empty shot71)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker408 l0)
  (shaker-level shaker408 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail197 ingredient423)
  (cocktail-part2 cocktail197 ingredient358)
)
 (:goal
  (and
      (contains shot90 cocktail197)
      (contains shot132 cocktail197)
)))
