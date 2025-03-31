(define (problem prob)
 (:domain barman)
 (:objects 
      shaker38 - shaker
      left right - hand
      shot86 - shot
      ingredient337 ingredient193 - ingredient
      cocktail1 - cocktail
      dispenser327 dispenser44 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker38)
  (ontable shot86)
  (dispenses dispenser327 ingredient337)
  (dispenses dispenser44 ingredient193)
  (clean shaker38)
  (clean shot86)
  (empty shaker38)
  (empty shot86)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker38 l0)
  (shaker-level shaker38 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient337)
  (cocktail-part2 cocktail1 ingredient193)
)
 (:goal
  (and
      (contains shot86 cocktail1)
)))
