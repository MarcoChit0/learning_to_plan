(define (problem prob)
 (:domain barman)
 (:objects 
      shaker34 - shaker
      left right - hand
      shot198 shot471 shot218 - shot
      ingredient260 ingredient370 ingredient101 ingredient475 - ingredient
      cocktail136 - cocktail
      dispenser472 dispenser264 dispenser128 dispenser28 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker34)
  (ontable shot198)
  (ontable shot471)
  (ontable shot218)
  (dispenses dispenser472 ingredient260)
  (dispenses dispenser264 ingredient370)
  (dispenses dispenser128 ingredient101)
  (dispenses dispenser28 ingredient475)
  (clean shaker34)
  (clean shot198)
  (clean shot471)
  (clean shot218)
  (empty shaker34)
  (empty shot198)
  (empty shot471)
  (empty shot218)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker34 l0)
  (shaker-level shaker34 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail136 ingredient370)
  (cocktail-part2 cocktail136 ingredient260)
)
 (:goal
  (and
      (contains shot198 cocktail136)
      (contains shot471 ingredient475)
)))
