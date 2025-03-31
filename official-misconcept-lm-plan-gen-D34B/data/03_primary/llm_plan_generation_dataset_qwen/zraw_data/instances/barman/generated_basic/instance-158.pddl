(define (problem prob)
 (:domain barman)
 (:objects 
      shaker45 - shaker
      left right - hand
      shot9 shot59 - shot
      ingredient344 ingredient140 ingredient428 - ingredient
      cocktail1 - cocktail
      dispenser253 dispenser194 dispenser106 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker45)
  (ontable shot9)
  (ontable shot59)
  (dispenses dispenser253 ingredient344)
  (dispenses dispenser194 ingredient140)
  (dispenses dispenser106 ingredient428)
  (clean shaker45)
  (clean shot9)
  (clean shot59)
  (empty shaker45)
  (empty shot9)
  (empty shot59)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker45 l0)
  (shaker-level shaker45 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient428)
  (cocktail-part2 cocktail1 ingredient140)
)
 (:goal
  (and
      (contains shot9 cocktail1)
)))
