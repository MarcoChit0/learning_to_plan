(define (problem prob)
 (:domain barman)
 (:objects 
      shaker406 - shaker
      left right - hand
      shot299 - shot
      ingredient431 ingredient94 ingredient387 - ingredient
      cocktail54 - cocktail
      dispenser408 dispenser360 dispenser482 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker406)
  (ontable shot299)
  (dispenses dispenser408 ingredient431)
  (dispenses dispenser360 ingredient94)
  (dispenses dispenser482 ingredient387)
  (clean shaker406)
  (clean shot299)
  (empty shaker406)
  (empty shot299)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker406 l0)
  (shaker-level shaker406 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail54 ingredient431)
  (cocktail-part2 cocktail54 ingredient94)
)
 (:goal
  (and
      (contains shot299 cocktail54)
)))
