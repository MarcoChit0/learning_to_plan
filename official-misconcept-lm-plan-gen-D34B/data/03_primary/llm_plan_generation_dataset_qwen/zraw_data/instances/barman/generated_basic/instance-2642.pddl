(define (problem prob)
 (:domain barman)
 (:objects 
      shaker73 - shaker
      left right - hand
      shot35 shot77 - shot
      ingredient173 ingredient444 ingredient66 - ingredient
      cocktail204 - cocktail
      dispenser120 dispenser338 dispenser393 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker73)
  (ontable shot35)
  (ontable shot77)
  (dispenses dispenser120 ingredient173)
  (dispenses dispenser338 ingredient444)
  (dispenses dispenser393 ingredient66)
  (clean shaker73)
  (clean shot35)
  (clean shot77)
  (empty shaker73)
  (empty shot35)
  (empty shot77)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker73 l0)
  (shaker-level shaker73 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail204 ingredient66)
  (cocktail-part2 cocktail204 ingredient173)
)
 (:goal
  (and
      (contains shot35 cocktail204)
)))
