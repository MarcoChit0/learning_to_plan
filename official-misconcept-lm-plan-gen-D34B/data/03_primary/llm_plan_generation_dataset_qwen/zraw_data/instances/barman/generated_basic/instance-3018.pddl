(define (problem prob)
 (:domain barman)
 (:objects 
      shaker244 - shaker
      left right - hand
      shot486 shot312 - shot
      ingredient406 ingredient102 ingredient325 - ingredient
      cocktail499 - cocktail
      dispenser463 dispenser464 dispenser297 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker244)
  (ontable shot486)
  (ontable shot312)
  (dispenses dispenser463 ingredient406)
  (dispenses dispenser464 ingredient102)
  (dispenses dispenser297 ingredient325)
  (clean shaker244)
  (clean shot486)
  (clean shot312)
  (empty shaker244)
  (empty shot486)
  (empty shot312)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker244 l0)
  (shaker-level shaker244 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail499 ingredient325)
  (cocktail-part2 cocktail499 ingredient102)
)
 (:goal
  (and
      (contains shot486 cocktail499)
)))
