(define (problem prob)
 (:domain barman)
 (:objects 
      shaker215 - shaker
      left right - hand
      shot469 - shot
      ingredient90 ingredient429 ingredient331 - ingredient
      cocktail1 - cocktail
      dispenser339 dispenser390 dispenser139 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker215)
  (ontable shot469)
  (dispenses dispenser339 ingredient90)
  (dispenses dispenser390 ingredient429)
  (dispenses dispenser139 ingredient331)
  (clean shaker215)
  (clean shot469)
  (empty shaker215)
  (empty shot469)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker215 l0)
  (shaker-level shaker215 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient90)
  (cocktail-part2 cocktail1 ingredient429)
)
 (:goal
  (and
      (contains shot469 cocktail1)
)))
