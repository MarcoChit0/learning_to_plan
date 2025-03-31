(define (problem prob)
 (:domain barman)
 (:objects 
      shaker103 - shaker
      left right - hand
      shot72 - shot
      ingredient373 ingredient100 - ingredient
      cocktail461 - cocktail
      dispenser276 dispenser375 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker103)
  (ontable shot72)
  (dispenses dispenser276 ingredient373)
  (dispenses dispenser375 ingredient100)
  (clean shaker103)
  (clean shot72)
  (empty shaker103)
  (empty shot72)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker103 l0)
  (shaker-level shaker103 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail461 ingredient100)
  (cocktail-part2 cocktail461 ingredient373)
)
 (:goal
  (and
      (contains shot72 cocktail461)
)))
