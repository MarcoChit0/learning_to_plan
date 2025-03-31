(define (problem prob)
 (:domain barman)
 (:objects 
      shaker474 - shaker
      left right - hand
      shot256 shot494 - shot
      ingredient66 ingredient392 - ingredient
      cocktail365 - cocktail
      dispenser58 dispenser126 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker474)
  (ontable shot256)
  (ontable shot494)
  (dispenses dispenser58 ingredient66)
  (dispenses dispenser126 ingredient392)
  (clean shaker474)
  (clean shot256)
  (clean shot494)
  (empty shaker474)
  (empty shot256)
  (empty shot494)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker474 l0)
  (shaker-level shaker474 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail365 ingredient66)
  (cocktail-part2 cocktail365 ingredient392)
)
 (:goal
  (and
      (contains shot256 cocktail365)
)))
