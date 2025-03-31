(define (problem prob)
 (:domain barman)
 (:objects 
      shaker271 - shaker
      left right - hand
      shot248 shot70 - shot
      ingredient375 ingredient370 - ingredient
      cocktail108 - cocktail
      dispenser363 dispenser384 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker271)
  (ontable shot248)
  (ontable shot70)
  (dispenses dispenser363 ingredient375)
  (dispenses dispenser384 ingredient370)
  (clean shaker271)
  (clean shot248)
  (clean shot70)
  (empty shaker271)
  (empty shot248)
  (empty shot70)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker271 l0)
  (shaker-level shaker271 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail108 ingredient370)
  (cocktail-part2 cocktail108 ingredient375)
)
 (:goal
  (and
      (contains shot248 cocktail108)
)))
