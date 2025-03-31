(define (problem prob)
 (:domain barman)
 (:objects 
      shaker363 - shaker
      left right - hand
      shot311 shot135 - shot
      ingredient426 ingredient448 - ingredient
      cocktail476 - cocktail
      dispenser262 dispenser436 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker363)
  (ontable shot311)
  (ontable shot135)
  (dispenses dispenser262 ingredient426)
  (dispenses dispenser436 ingredient448)
  (clean shaker363)
  (clean shot311)
  (clean shot135)
  (empty shaker363)
  (empty shot311)
  (empty shot135)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker363 l0)
  (shaker-level shaker363 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail476 ingredient426)
  (cocktail-part2 cocktail476 ingredient448)
)
 (:goal
  (and
      (contains shot311 cocktail476)
)))
