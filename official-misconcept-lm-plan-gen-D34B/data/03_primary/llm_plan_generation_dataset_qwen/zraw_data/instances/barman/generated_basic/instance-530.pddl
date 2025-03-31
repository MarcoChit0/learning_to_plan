(define (problem prob)
 (:domain barman)
 (:objects 
      shaker418 - shaker
      left right - hand
      shot327 shot9 - shot
      ingredient357 ingredient497 - ingredient
      cocktail1 - cocktail
      dispenser132 dispenser93 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker418)
  (ontable shot327)
  (ontable shot9)
  (dispenses dispenser132 ingredient357)
  (dispenses dispenser93 ingredient497)
  (clean shaker418)
  (clean shot327)
  (clean shot9)
  (empty shaker418)
  (empty shot327)
  (empty shot9)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker418 l0)
  (shaker-level shaker418 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient497)
  (cocktail-part2 cocktail1 ingredient357)
)
 (:goal
  (and
      (contains shot327 cocktail1)
)))
