(define (problem prob)
 (:domain barman)
 (:objects 
      shaker239 - shaker
      left right - hand
      shot174 - shot
      ingredient311 ingredient180 - ingredient
      cocktail109 - cocktail
      dispenser232 dispenser19 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker239)
  (ontable shot174)
  (dispenses dispenser232 ingredient311)
  (dispenses dispenser19 ingredient180)
  (clean shaker239)
  (clean shot174)
  (empty shaker239)
  (empty shot174)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker239 l0)
  (shaker-level shaker239 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail109 ingredient180)
  (cocktail-part2 cocktail109 ingredient311)
)
 (:goal
  (and
      (contains shot174 cocktail109)
)))
