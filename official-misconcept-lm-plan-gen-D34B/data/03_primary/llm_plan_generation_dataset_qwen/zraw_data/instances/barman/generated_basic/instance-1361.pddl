(define (problem prob)
 (:domain barman)
 (:objects 
      shaker309 - shaker
      left right - hand
      shot53 - shot
      ingredient53 ingredient213 ingredient27 - ingredient
      cocktail213 - cocktail
      dispenser257 dispenser312 dispenser218 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker309)
  (ontable shot53)
  (dispenses dispenser257 ingredient53)
  (dispenses dispenser312 ingredient213)
  (dispenses dispenser218 ingredient27)
  (clean shaker309)
  (clean shot53)
  (empty shaker309)
  (empty shot53)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker309 l0)
  (shaker-level shaker309 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail213 ingredient213)
  (cocktail-part2 cocktail213 ingredient27)
)
 (:goal
  (and
      (contains shot53 cocktail213)
)))
