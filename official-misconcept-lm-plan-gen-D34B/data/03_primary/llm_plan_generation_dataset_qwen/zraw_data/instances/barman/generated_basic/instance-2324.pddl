(define (problem prob)
 (:domain barman)
 (:objects 
      shaker441 - shaker
      left right - hand
      shot42 - shot
      ingredient53 ingredient401 - ingredient
      cocktail356 - cocktail
      dispenser138 dispenser106 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker441)
  (ontable shot42)
  (dispenses dispenser138 ingredient53)
  (dispenses dispenser106 ingredient401)
  (clean shaker441)
  (clean shot42)
  (empty shaker441)
  (empty shot42)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker441 l0)
  (shaker-level shaker441 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail356 ingredient401)
  (cocktail-part2 cocktail356 ingredient53)
)
 (:goal
  (and
      (contains shot42 cocktail356)
)))
