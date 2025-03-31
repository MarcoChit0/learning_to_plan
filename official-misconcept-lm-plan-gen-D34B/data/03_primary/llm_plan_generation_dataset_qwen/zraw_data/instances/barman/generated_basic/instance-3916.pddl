(define (problem prob)
 (:domain barman)
 (:objects 
      shaker478 - shaker
      left right - hand
      shot253 - shot
      ingredient450 ingredient329 ingredient351 ingredient5 - ingredient
      cocktail20 - cocktail
      dispenser0 dispenser374 dispenser458 dispenser170 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker478)
  (ontable shot253)
  (dispenses dispenser0 ingredient450)
  (dispenses dispenser374 ingredient329)
  (dispenses dispenser458 ingredient351)
  (dispenses dispenser170 ingredient5)
  (clean shaker478)
  (clean shot253)
  (empty shaker478)
  (empty shot253)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker478 l0)
  (shaker-level shaker478 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail20 ingredient351)
  (cocktail-part2 cocktail20 ingredient450)
)
 (:goal
  (and
      (contains shot253 cocktail20)
)))
