(define (problem prob)
 (:domain barman)
 (:objects 
      shaker336 - shaker
      left right - hand
      shot381 shot73 - shot
      ingredient101 ingredient207 ingredient373 - ingredient
      cocktail1 - cocktail
      dispenser48 dispenser400 dispenser56 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker336)
  (ontable shot381)
  (ontable shot73)
  (dispenses dispenser48 ingredient101)
  (dispenses dispenser400 ingredient207)
  (dispenses dispenser56 ingredient373)
  (clean shaker336)
  (clean shot381)
  (clean shot73)
  (empty shaker336)
  (empty shot381)
  (empty shot73)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker336 l0)
  (shaker-level shaker336 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient373)
  (cocktail-part2 cocktail1 ingredient207)
)
 (:goal
  (and
      (contains shot381 cocktail1)
)))
