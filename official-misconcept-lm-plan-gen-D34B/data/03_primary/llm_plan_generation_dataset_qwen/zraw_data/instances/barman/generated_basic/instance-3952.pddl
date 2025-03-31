(define (problem prob)
 (:domain barman)
 (:objects 
      shaker78 - shaker
      left right - hand
      shot464 shot139 - shot
      ingredient66 ingredient483 ingredient418 ingredient267 - ingredient
      cocktail14 - cocktail
      dispenser430 dispenser400 dispenser147 dispenser312 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker78)
  (ontable shot464)
  (ontable shot139)
  (dispenses dispenser430 ingredient66)
  (dispenses dispenser400 ingredient483)
  (dispenses dispenser147 ingredient418)
  (dispenses dispenser312 ingredient267)
  (clean shaker78)
  (clean shot464)
  (clean shot139)
  (empty shaker78)
  (empty shot464)
  (empty shot139)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker78 l0)
  (shaker-level shaker78 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail14 ingredient418)
  (cocktail-part2 cocktail14 ingredient66)
)
 (:goal
  (and
      (contains shot464 cocktail14)
)))
