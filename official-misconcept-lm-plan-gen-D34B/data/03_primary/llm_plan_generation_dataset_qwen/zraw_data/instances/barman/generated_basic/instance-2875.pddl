(define (problem prob)
 (:domain barman)
 (:objects 
      shaker253 - shaker
      left right - hand
      shot297 shot459 shot211 - shot
      ingredient27 ingredient67 - ingredient
      cocktail491 - cocktail
      dispenser297 dispenser440 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker253)
  (ontable shot297)
  (ontable shot459)
  (ontable shot211)
  (dispenses dispenser297 ingredient27)
  (dispenses dispenser440 ingredient67)
  (clean shaker253)
  (clean shot297)
  (clean shot459)
  (clean shot211)
  (empty shaker253)
  (empty shot297)
  (empty shot459)
  (empty shot211)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker253 l0)
  (shaker-level shaker253 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail491 ingredient27)
  (cocktail-part2 cocktail491 ingredient67)
)
 (:goal
  (and
      (contains shot297 cocktail491)
      (contains shot459 cocktail491)
)))
