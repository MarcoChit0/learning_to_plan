(define (problem prob)
 (:domain barman)
 (:objects 
      shaker412 - shaker
      left right - hand
      shot459 - shot
      ingredient335 ingredient342 ingredient74 ingredient67 - ingredient
      cocktail229 - cocktail
      dispenser297 dispenser452 dispenser459 dispenser342 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker412)
  (ontable shot459)
  (dispenses dispenser297 ingredient335)
  (dispenses dispenser452 ingredient342)
  (dispenses dispenser459 ingredient74)
  (dispenses dispenser342 ingredient67)
  (clean shaker412)
  (clean shot459)
  (empty shaker412)
  (empty shot459)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker412 l0)
  (shaker-level shaker412 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail229 ingredient67)
  (cocktail-part2 cocktail229 ingredient74)
)
 (:goal
  (and
      (contains shot459 cocktail229)
)))
