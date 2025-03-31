(define (problem prob)
 (:domain barman)
 (:objects 
      shaker434 - shaker
      left right - hand
      shot218 - shot
      ingredient211 ingredient180 - ingredient
      cocktail244 - cocktail
      dispenser4 dispenser478 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker434)
  (ontable shot218)
  (dispenses dispenser4 ingredient211)
  (dispenses dispenser478 ingredient180)
  (clean shaker434)
  (clean shot218)
  (empty shaker434)
  (empty shot218)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker434 l0)
  (shaker-level shaker434 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail244 ingredient180)
  (cocktail-part2 cocktail244 ingredient211)
)
 (:goal
  (and
      (contains shot218 cocktail244)
)))
