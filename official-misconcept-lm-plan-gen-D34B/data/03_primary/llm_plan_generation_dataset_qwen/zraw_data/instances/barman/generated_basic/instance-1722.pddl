(define (problem prob)
 (:domain barman)
 (:objects 
      shaker74 - shaker
      left right - hand
      shot80 shot96 - shot
      ingredient131 ingredient171 ingredient110 ingredient66 - ingredient
      cocktail82 - cocktail
      dispenser487 dispenser488 dispenser422 dispenser321 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker74)
  (ontable shot80)
  (ontable shot96)
  (dispenses dispenser487 ingredient131)
  (dispenses dispenser488 ingredient171)
  (dispenses dispenser422 ingredient110)
  (dispenses dispenser321 ingredient66)
  (clean shaker74)
  (clean shot80)
  (clean shot96)
  (empty shaker74)
  (empty shot80)
  (empty shot96)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker74 l0)
  (shaker-level shaker74 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail82 ingredient171)
  (cocktail-part2 cocktail82 ingredient66)
)
 (:goal
  (and
      (contains shot80 cocktail82)
)))
