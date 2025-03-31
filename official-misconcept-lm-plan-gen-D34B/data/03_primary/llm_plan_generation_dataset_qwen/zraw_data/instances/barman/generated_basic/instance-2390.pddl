(define (problem prob)
 (:domain barman)
 (:objects 
      shaker379 - shaker
      left right - hand
      shot9 shot311 - shot
      ingredient101 ingredient121 ingredient300 - ingredient
      cocktail3 - cocktail
      dispenser24 dispenser215 dispenser254 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker379)
  (ontable shot9)
  (ontable shot311)
  (dispenses dispenser24 ingredient101)
  (dispenses dispenser215 ingredient121)
  (dispenses dispenser254 ingredient300)
  (clean shaker379)
  (clean shot9)
  (clean shot311)
  (empty shaker379)
  (empty shot9)
  (empty shot311)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker379 l0)
  (shaker-level shaker379 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail3 ingredient101)
  (cocktail-part2 cocktail3 ingredient121)
)
 (:goal
  (and
      (contains shot9 cocktail3)
)))
