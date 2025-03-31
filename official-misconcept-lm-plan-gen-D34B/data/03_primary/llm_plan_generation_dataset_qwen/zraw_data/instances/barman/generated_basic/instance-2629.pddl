(define (problem prob)
 (:domain barman)
 (:objects 
      shaker28 - shaker
      left right - hand
      shot483 shot311 - shot
      ingredient64 ingredient41 ingredient109 - ingredient
      cocktail390 - cocktail
      dispenser497 dispenser196 dispenser73 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker28)
  (ontable shot483)
  (ontable shot311)
  (dispenses dispenser497 ingredient64)
  (dispenses dispenser196 ingredient41)
  (dispenses dispenser73 ingredient109)
  (clean shaker28)
  (clean shot483)
  (clean shot311)
  (empty shaker28)
  (empty shot483)
  (empty shot311)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker28 l0)
  (shaker-level shaker28 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail390 ingredient64)
  (cocktail-part2 cocktail390 ingredient41)
)
 (:goal
  (and
      (contains shot483 cocktail390)
)))
