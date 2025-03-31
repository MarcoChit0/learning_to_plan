(define (problem prob)
 (:domain barman)
 (:objects 
      shaker196 - shaker
      left right - hand
      shot494 shot304 - shot
      ingredient214 ingredient302 ingredient487 ingredient272 - ingredient
      cocktail358 - cocktail
      dispenser233 dispenser460 dispenser222 dispenser29 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker196)
  (ontable shot494)
  (ontable shot304)
  (dispenses dispenser233 ingredient214)
  (dispenses dispenser460 ingredient302)
  (dispenses dispenser222 ingredient487)
  (dispenses dispenser29 ingredient272)
  (clean shaker196)
  (clean shot494)
  (clean shot304)
  (empty shaker196)
  (empty shot494)
  (empty shot304)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker196 l0)
  (shaker-level shaker196 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail358 ingredient214)
  (cocktail-part2 cocktail358 ingredient272)
)
 (:goal
  (and
      (contains shot494 cocktail358)
)))
