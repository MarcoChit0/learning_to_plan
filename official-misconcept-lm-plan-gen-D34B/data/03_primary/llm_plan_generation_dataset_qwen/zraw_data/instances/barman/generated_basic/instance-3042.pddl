(define (problem prob)
 (:domain barman)
 (:objects 
      shaker491 - shaker
      left right - hand
      shot134 shot475 - shot
      ingredient369 ingredient330 ingredient107 - ingredient
      cocktail236 - cocktail
      dispenser244 dispenser139 dispenser363 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker491)
  (ontable shot134)
  (ontable shot475)
  (dispenses dispenser244 ingredient369)
  (dispenses dispenser139 ingredient330)
  (dispenses dispenser363 ingredient107)
  (clean shaker491)
  (clean shot134)
  (clean shot475)
  (empty shaker491)
  (empty shot134)
  (empty shot475)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker491 l0)
  (shaker-level shaker491 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail236 ingredient107)
  (cocktail-part2 cocktail236 ingredient369)
)
 (:goal
  (and
      (contains shot134 cocktail236)
)))
