(define (problem prob)
 (:domain barman)
 (:objects 
      shaker113 - shaker
      left right - hand
      shot42 shot275 - shot
      ingredient188 ingredient494 ingredient304 ingredient110 - ingredient
      cocktail494 - cocktail
      dispenser268 dispenser172 dispenser393 dispenser417 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker113)
  (ontable shot42)
  (ontable shot275)
  (dispenses dispenser268 ingredient188)
  (dispenses dispenser172 ingredient494)
  (dispenses dispenser393 ingredient304)
  (dispenses dispenser417 ingredient110)
  (clean shaker113)
  (clean shot42)
  (clean shot275)
  (empty shaker113)
  (empty shot42)
  (empty shot275)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker113 l0)
  (shaker-level shaker113 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail494 ingredient494)
  (cocktail-part2 cocktail494 ingredient110)
)
 (:goal
  (and
      (contains shot42 cocktail494)
)))
