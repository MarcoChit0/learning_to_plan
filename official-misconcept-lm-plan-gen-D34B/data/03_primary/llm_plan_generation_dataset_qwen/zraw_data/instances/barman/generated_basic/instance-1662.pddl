(define (problem prob)
 (:domain barman)
 (:objects 
      shaker84 - shaker
      left right - hand
      shot185 shot377 - shot
      ingredient293 ingredient71 ingredient388 ingredient113 - ingredient
      cocktail12 - cocktail
      dispenser491 dispenser380 dispenser292 dispenser57 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker84)
  (ontable shot185)
  (ontable shot377)
  (dispenses dispenser491 ingredient293)
  (dispenses dispenser380 ingredient71)
  (dispenses dispenser292 ingredient388)
  (dispenses dispenser57 ingredient113)
  (clean shaker84)
  (clean shot185)
  (clean shot377)
  (empty shaker84)
  (empty shot185)
  (empty shot377)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker84 l0)
  (shaker-level shaker84 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail12 ingredient113)
  (cocktail-part2 cocktail12 ingredient71)
)
 (:goal
  (and
      (contains shot185 cocktail12)
)))
