(define (problem prob)
 (:domain barman)
 (:objects 
      shaker239 - shaker
      left right - hand
      shot131 - shot
      ingredient123 ingredient302 ingredient121 ingredient18 - ingredient
      cocktail82 - cocktail
      dispenser401 dispenser468 dispenser257 dispenser388 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker239)
  (ontable shot131)
  (dispenses dispenser401 ingredient123)
  (dispenses dispenser468 ingredient302)
  (dispenses dispenser257 ingredient121)
  (dispenses dispenser388 ingredient18)
  (clean shaker239)
  (clean shot131)
  (empty shaker239)
  (empty shot131)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker239 l0)
  (shaker-level shaker239 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail82 ingredient302)
  (cocktail-part2 cocktail82 ingredient18)
)
 (:goal
  (and
      (contains shot131 cocktail82)
)))
