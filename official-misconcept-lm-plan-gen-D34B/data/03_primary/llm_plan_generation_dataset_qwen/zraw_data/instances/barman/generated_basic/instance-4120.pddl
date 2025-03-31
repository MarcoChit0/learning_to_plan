(define (problem prob)
 (:domain barman)
 (:objects 
      shaker55 - shaker
      left right - hand
      shot135 shot123 - shot
      ingredient256 ingredient103 ingredient393 - ingredient
      cocktail419 - cocktail
      dispenser100 dispenser441 dispenser491 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker55)
  (ontable shot135)
  (ontable shot123)
  (dispenses dispenser100 ingredient256)
  (dispenses dispenser441 ingredient103)
  (dispenses dispenser491 ingredient393)
  (clean shaker55)
  (clean shot135)
  (clean shot123)
  (empty shaker55)
  (empty shot135)
  (empty shot123)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker55 l0)
  (shaker-level shaker55 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail419 ingredient103)
  (cocktail-part2 cocktail419 ingredient256)
)
 (:goal
  (and
      (contains shot135 cocktail419)
)))
