(define (problem prob)
 (:domain barman)
 (:objects 
      shaker211 - shaker
      left right - hand
      shot180 - shot
      ingredient495 ingredient302 - ingredient
      cocktail453 - cocktail
      dispenser334 dispenser113 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker211)
  (ontable shot180)
  (dispenses dispenser334 ingredient495)
  (dispenses dispenser113 ingredient302)
  (clean shaker211)
  (clean shot180)
  (empty shaker211)
  (empty shot180)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker211 l0)
  (shaker-level shaker211 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail453 ingredient495)
  (cocktail-part2 cocktail453 ingredient302)
)
 (:goal
  (and
      (contains shot180 cocktail453)
)))
