(define (problem prob)
 (:domain barman)
 (:objects 
      shaker253 - shaker
      left right - hand
      shot311 - shot
      ingredient304 ingredient401 - ingredient
      cocktail341 - cocktail
      dispenser432 dispenser130 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker253)
  (ontable shot311)
  (dispenses dispenser432 ingredient304)
  (dispenses dispenser130 ingredient401)
  (clean shaker253)
  (clean shot311)
  (empty shaker253)
  (empty shot311)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker253 l0)
  (shaker-level shaker253 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail341 ingredient401)
  (cocktail-part2 cocktail341 ingredient304)
)
 (:goal
  (and
      (contains shot311 cocktail341)
)))
