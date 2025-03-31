(define (problem prob)
 (:domain barman)
 (:objects 
      shaker357 - shaker
      left right - hand
      shot72 - shot
      ingredient467 ingredient375 - ingredient
      cocktail344 - cocktail
      dispenser439 dispenser286 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker357)
  (ontable shot72)
  (dispenses dispenser439 ingredient467)
  (dispenses dispenser286 ingredient375)
  (clean shaker357)
  (clean shot72)
  (empty shaker357)
  (empty shot72)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker357 l0)
  (shaker-level shaker357 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail344 ingredient375)
  (cocktail-part2 cocktail344 ingredient467)
)
 (:goal
  (and
      (contains shot72 cocktail344)
)))
