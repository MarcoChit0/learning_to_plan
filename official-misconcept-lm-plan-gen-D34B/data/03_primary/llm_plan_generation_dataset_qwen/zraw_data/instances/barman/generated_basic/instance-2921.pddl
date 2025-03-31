(define (problem prob)
 (:domain barman)
 (:objects 
      shaker472 - shaker
      left right - hand
      shot53 - shot
      ingredient90 ingredient363 - ingredient
      cocktail237 - cocktail
      dispenser75 dispenser320 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker472)
  (ontable shot53)
  (dispenses dispenser75 ingredient90)
  (dispenses dispenser320 ingredient363)
  (clean shaker472)
  (clean shot53)
  (empty shaker472)
  (empty shot53)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker472 l0)
  (shaker-level shaker472 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail237 ingredient90)
  (cocktail-part2 cocktail237 ingredient363)
)
 (:goal
  (and
      (contains shot53 cocktail237)
)))
