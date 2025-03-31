(define (problem prob)
 (:domain barman)
 (:objects 
      shaker76 - shaker
      left right - hand
      shot381 shot273 shot256 - shot
      ingredient277 ingredient330 - ingredient
      cocktail30 - cocktail
      dispenser430 dispenser393 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker76)
  (ontable shot381)
  (ontable shot273)
  (ontable shot256)
  (dispenses dispenser430 ingredient277)
  (dispenses dispenser393 ingredient330)
  (clean shaker76)
  (clean shot381)
  (clean shot273)
  (clean shot256)
  (empty shaker76)
  (empty shot381)
  (empty shot273)
  (empty shot256)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker76 l0)
  (shaker-level shaker76 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail30 ingredient277)
  (cocktail-part2 cocktail30 ingredient330)
)
 (:goal
  (and
      (contains shot381 cocktail30)
      (contains shot273 cocktail30)
)))
