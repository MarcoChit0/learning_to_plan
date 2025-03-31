(define (problem prob)
 (:domain barman)
 (:objects 
      shaker178 - shaker
      left right - hand
      shot320 shot193 - shot
      ingredient66 ingredient297 - ingredient
      cocktail110 - cocktail
      dispenser7 dispenser471 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker178)
  (ontable shot320)
  (ontable shot193)
  (dispenses dispenser7 ingredient66)
  (dispenses dispenser471 ingredient297)
  (clean shaker178)
  (clean shot320)
  (clean shot193)
  (empty shaker178)
  (empty shot320)
  (empty shot193)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker178 l0)
  (shaker-level shaker178 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail110 ingredient66)
  (cocktail-part2 cocktail110 ingredient297)
)
 (:goal
  (and
      (contains shot320 cocktail110)
)))
