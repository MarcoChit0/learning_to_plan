(define (problem prob)
 (:domain barman)
 (:objects 
      shaker308 - shaker
      left right - hand
      shot387 shot3 - shot
      ingredient329 ingredient60 - ingredient
      cocktail1 - cocktail
      dispenser453 dispenser115 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker308)
  (ontable shot387)
  (ontable shot3)
  (dispenses dispenser453 ingredient329)
  (dispenses dispenser115 ingredient60)
  (clean shaker308)
  (clean shot387)
  (clean shot3)
  (empty shaker308)
  (empty shot387)
  (empty shot3)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker308 l0)
  (shaker-level shaker308 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient329)
  (cocktail-part2 cocktail1 ingredient60)
)
 (:goal
  (and
      (contains shot387 cocktail1)
)))
