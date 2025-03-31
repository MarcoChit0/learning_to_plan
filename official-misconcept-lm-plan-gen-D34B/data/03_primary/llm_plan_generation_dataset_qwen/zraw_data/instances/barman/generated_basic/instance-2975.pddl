(define (problem prob)
 (:domain barman)
 (:objects 
      shaker463 - shaker
      left right - hand
      shot168 - shot
      ingredient454 ingredient208 - ingredient
      cocktail263 - cocktail
      dispenser214 dispenser463 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker463)
  (ontable shot168)
  (dispenses dispenser214 ingredient454)
  (dispenses dispenser463 ingredient208)
  (clean shaker463)
  (clean shot168)
  (empty shaker463)
  (empty shot168)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker463 l0)
  (shaker-level shaker463 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail263 ingredient208)
  (cocktail-part2 cocktail263 ingredient454)
)
 (:goal
  (and
      (contains shot168 cocktail263)
)))
