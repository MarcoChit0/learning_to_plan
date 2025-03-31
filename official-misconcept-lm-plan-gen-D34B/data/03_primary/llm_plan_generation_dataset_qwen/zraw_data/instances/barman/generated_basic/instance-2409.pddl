(define (problem prob)
 (:domain barman)
 (:objects 
      shaker415 - shaker
      left right - hand
      shot72 - shot
      ingredient346 ingredient182 - ingredient
      cocktail319 - cocktail
      dispenser462 dispenser38 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker415)
  (ontable shot72)
  (dispenses dispenser462 ingredient346)
  (dispenses dispenser38 ingredient182)
  (clean shaker415)
  (clean shot72)
  (empty shaker415)
  (empty shot72)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker415 l0)
  (shaker-level shaker415 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail319 ingredient346)
  (cocktail-part2 cocktail319 ingredient182)
)
 (:goal
  (and
      (contains shot72 cocktail319)
)))
