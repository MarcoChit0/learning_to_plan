(define (problem prob)
 (:domain barman)
 (:objects 
      shaker14 - shaker
      left right - hand
      shot352 shot335 - shot
      ingredient75 ingredient320 - ingredient
      cocktail163 - cocktail
      dispenser156 dispenser220 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker14)
  (ontable shot352)
  (ontable shot335)
  (dispenses dispenser156 ingredient75)
  (dispenses dispenser220 ingredient320)
  (clean shaker14)
  (clean shot352)
  (clean shot335)
  (empty shaker14)
  (empty shot352)
  (empty shot335)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker14 l0)
  (shaker-level shaker14 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail163 ingredient75)
  (cocktail-part2 cocktail163 ingredient320)
)
 (:goal
  (and
      (contains shot352 cocktail163)
)))
