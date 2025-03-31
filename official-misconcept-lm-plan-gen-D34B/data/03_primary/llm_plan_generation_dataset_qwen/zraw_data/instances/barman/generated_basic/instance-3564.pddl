(define (problem prob)
 (:domain barman)
 (:objects 
      shaker155 - shaker
      left right - hand
      shot418 shot300 - shot
      ingredient360 ingredient235 ingredient497 ingredient9 - ingredient
      cocktail200 - cocktail
      dispenser443 dispenser70 dispenser427 dispenser280 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker155)
  (ontable shot418)
  (ontable shot300)
  (dispenses dispenser443 ingredient360)
  (dispenses dispenser70 ingredient235)
  (dispenses dispenser427 ingredient497)
  (dispenses dispenser280 ingredient9)
  (clean shaker155)
  (clean shot418)
  (clean shot300)
  (empty shaker155)
  (empty shot418)
  (empty shot300)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker155 l0)
  (shaker-level shaker155 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail200 ingredient235)
  (cocktail-part2 cocktail200 ingredient360)
)
 (:goal
  (and
      (contains shot418 cocktail200)
)))
