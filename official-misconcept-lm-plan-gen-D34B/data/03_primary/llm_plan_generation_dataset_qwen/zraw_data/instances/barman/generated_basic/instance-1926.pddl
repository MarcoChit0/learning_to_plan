(define (problem prob)
 (:domain barman)
 (:objects 
      shaker69 - shaker
      left right - hand
      shot127 - shot
      ingredient14 ingredient293 - ingredient
      cocktail239 - cocktail
      dispenser54 dispenser232 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker69)
  (ontable shot127)
  (dispenses dispenser54 ingredient14)
  (dispenses dispenser232 ingredient293)
  (clean shaker69)
  (clean shot127)
  (empty shaker69)
  (empty shot127)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker69 l0)
  (shaker-level shaker69 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail239 ingredient293)
  (cocktail-part2 cocktail239 ingredient14)
)
 (:goal
  (and
      (contains shot127 cocktail239)
)))
