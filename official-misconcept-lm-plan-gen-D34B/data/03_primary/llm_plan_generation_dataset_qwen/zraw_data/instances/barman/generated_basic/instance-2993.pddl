(define (problem prob)
 (:domain barman)
 (:objects 
      shaker125 - shaker
      left right - hand
      shot193 shot116 - shot
      ingredient79 ingredient404 ingredient348 ingredient354 - ingredient
      cocktail75 - cocktail
      dispenser287 dispenser264 dispenser486 dispenser343 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker125)
  (ontable shot193)
  (ontable shot116)
  (dispenses dispenser287 ingredient79)
  (dispenses dispenser264 ingredient404)
  (dispenses dispenser486 ingredient348)
  (dispenses dispenser343 ingredient354)
  (clean shaker125)
  (clean shot193)
  (clean shot116)
  (empty shaker125)
  (empty shot193)
  (empty shot116)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker125 l0)
  (shaker-level shaker125 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail75 ingredient354)
  (cocktail-part2 cocktail75 ingredient404)
)
 (:goal
  (and
      (contains shot193 cocktail75)
)))
