(define (problem prob)
 (:domain barman)
 (:objects 
      shaker485 - shaker
      left right - hand
      shot284 - shot
      ingredient66 ingredient410 - ingredient
      cocktail1 - cocktail
      dispenser16 dispenser201 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker485)
  (ontable shot284)
  (dispenses dispenser16 ingredient66)
  (dispenses dispenser201 ingredient410)
  (clean shaker485)
  (clean shot284)
  (empty shaker485)
  (empty shot284)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker485 l0)
  (shaker-level shaker485 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient410)
  (cocktail-part2 cocktail1 ingredient66)
)
 (:goal
  (and
      (contains shot284 cocktail1)
)))
