(define (problem prob)
 (:domain barman)
 (:objects 
      shaker315 - shaker
      left right - hand
      shot86 - shot
      ingredient468 ingredient78 ingredient478 - ingredient
      cocktail1 - cocktail
      dispenser403 dispenser486 dispenser36 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker315)
  (ontable shot86)
  (dispenses dispenser403 ingredient468)
  (dispenses dispenser486 ingredient78)
  (dispenses dispenser36 ingredient478)
  (clean shaker315)
  (clean shot86)
  (empty shaker315)
  (empty shot86)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker315 l0)
  (shaker-level shaker315 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient78)
  (cocktail-part2 cocktail1 ingredient468)
)
 (:goal
  (and
      (contains shot86 cocktail1)
)))
