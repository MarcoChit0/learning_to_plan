(define (problem prob)
 (:domain barman)
 (:objects 
      shaker385 - shaker
      left right - hand
      shot35 - shot
      ingredient101 ingredient311 ingredient4 ingredient31 - ingredient
      cocktail186 - cocktail
      dispenser268 dispenser182 dispenser321 dispenser430 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker385)
  (ontable shot35)
  (dispenses dispenser268 ingredient101)
  (dispenses dispenser182 ingredient311)
  (dispenses dispenser321 ingredient4)
  (dispenses dispenser430 ingredient31)
  (clean shaker385)
  (clean shot35)
  (empty shaker385)
  (empty shot35)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker385 l0)
  (shaker-level shaker385 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail186 ingredient4)
  (cocktail-part2 cocktail186 ingredient31)
)
 (:goal
  (and
      (contains shot35 cocktail186)
)))
