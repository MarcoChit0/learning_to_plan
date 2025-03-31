(define (problem prob)
 (:domain barman)
 (:objects 
      shaker113 - shaker
      left right - hand
      shot25 - shot
      ingredient218 ingredient35 ingredient464 ingredient107 - ingredient
      cocktail1 - cocktail
      dispenser343 dispenser297 dispenser456 dispenser64 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker113)
  (ontable shot25)
  (dispenses dispenser343 ingredient218)
  (dispenses dispenser297 ingredient35)
  (dispenses dispenser456 ingredient464)
  (dispenses dispenser64 ingredient107)
  (clean shaker113)
  (clean shot25)
  (empty shaker113)
  (empty shot25)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker113 l0)
  (shaker-level shaker113 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient35)
  (cocktail-part2 cocktail1 ingredient218)
)
 (:goal
  (and
      (contains shot25 cocktail1)
)))
