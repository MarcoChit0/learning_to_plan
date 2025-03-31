(define (problem prob)
 (:domain barman)
 (:objects 
      shaker409 - shaker
      left right - hand
      shot308 shot107 - shot
      ingredient302 ingredient140 ingredient314 ingredient282 - ingredient
      cocktail468 - cocktail
      dispenser253 dispenser248 dispenser3 dispenser385 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker409)
  (ontable shot308)
  (ontable shot107)
  (dispenses dispenser253 ingredient302)
  (dispenses dispenser248 ingredient140)
  (dispenses dispenser3 ingredient314)
  (dispenses dispenser385 ingredient282)
  (clean shaker409)
  (clean shot308)
  (clean shot107)
  (empty shaker409)
  (empty shot308)
  (empty shot107)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker409 l0)
  (shaker-level shaker409 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail468 ingredient302)
  (cocktail-part2 cocktail468 ingredient140)
)
 (:goal
  (and
      (contains shot308 cocktail468)
)))
