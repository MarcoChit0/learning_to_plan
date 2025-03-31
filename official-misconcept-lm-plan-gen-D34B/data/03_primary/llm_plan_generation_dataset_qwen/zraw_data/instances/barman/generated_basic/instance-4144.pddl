(define (problem prob)
 (:domain barman)
 (:objects 
      shaker494 - shaker
      left right - hand
      shot352 - shot
      ingredient82 ingredient98 - ingredient
      cocktail438 - cocktail
      dispenser329 dispenser78 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker494)
  (ontable shot352)
  (dispenses dispenser329 ingredient82)
  (dispenses dispenser78 ingredient98)
  (clean shaker494)
  (clean shot352)
  (empty shaker494)
  (empty shot352)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker494 l0)
  (shaker-level shaker494 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail438 ingredient98)
  (cocktail-part2 cocktail438 ingredient82)
)
 (:goal
  (and
      (contains shot352 cocktail438)
)))
