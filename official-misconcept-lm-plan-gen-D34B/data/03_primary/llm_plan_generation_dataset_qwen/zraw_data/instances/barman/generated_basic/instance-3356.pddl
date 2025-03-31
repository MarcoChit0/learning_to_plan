(define (problem prob)
 (:domain barman)
 (:objects 
      shaker329 - shaker
      left right - hand
      shot1 - shot
      ingredient91 ingredient95 - ingredient
      cocktail273 - cocktail
      dispenser199 dispenser429 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker329)
  (ontable shot1)
  (dispenses dispenser199 ingredient91)
  (dispenses dispenser429 ingredient95)
  (clean shaker329)
  (clean shot1)
  (empty shaker329)
  (empty shot1)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker329 l0)
  (shaker-level shaker329 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail273 ingredient95)
  (cocktail-part2 cocktail273 ingredient91)
)
 (:goal
  (and
      (contains shot1 cocktail273)
)))
