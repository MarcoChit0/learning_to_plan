(define (problem prob)
 (:domain barman)
 (:objects 
      shaker104 - shaker
      left right - hand
      shot348 - shot
      ingredient293 ingredient113 - ingredient
      cocktail32 - cocktail
      dispenser205 dispenser343 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker104)
  (ontable shot348)
  (dispenses dispenser205 ingredient293)
  (dispenses dispenser343 ingredient113)
  (clean shaker104)
  (clean shot348)
  (empty shaker104)
  (empty shot348)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker104 l0)
  (shaker-level shaker104 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail32 ingredient113)
  (cocktail-part2 cocktail32 ingredient293)
)
 (:goal
  (and
      (contains shot348 cocktail32)
)))
