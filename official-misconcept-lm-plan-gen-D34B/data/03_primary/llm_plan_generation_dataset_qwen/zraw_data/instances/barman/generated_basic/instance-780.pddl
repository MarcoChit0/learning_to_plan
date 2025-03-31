(define (problem prob)
 (:domain barman)
 (:objects 
      shaker233 - shaker
      left right - hand
      shot127 - shot
      ingredient40 ingredient128 - ingredient
      cocktail1 - cocktail
      dispenser191 dispenser20 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker233)
  (ontable shot127)
  (dispenses dispenser191 ingredient40)
  (dispenses dispenser20 ingredient128)
  (clean shaker233)
  (clean shot127)
  (empty shaker233)
  (empty shot127)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker233 l0)
  (shaker-level shaker233 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient128)
  (cocktail-part2 cocktail1 ingredient40)
)
 (:goal
  (and
      (contains shot127 cocktail1)
)))
