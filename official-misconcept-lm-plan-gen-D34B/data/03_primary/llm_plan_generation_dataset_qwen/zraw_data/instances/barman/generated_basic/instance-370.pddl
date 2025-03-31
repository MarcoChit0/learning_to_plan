(define (problem prob)
 (:domain barman)
 (:objects 
      shaker136 - shaker
      left right - hand
      shot119 - shot
      ingredient397 ingredient67 - ingredient
      cocktail1 - cocktail
      dispenser3 dispenser403 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker136)
  (ontable shot119)
  (dispenses dispenser3 ingredient397)
  (dispenses dispenser403 ingredient67)
  (clean shaker136)
  (clean shot119)
  (empty shaker136)
  (empty shot119)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker136 l0)
  (shaker-level shaker136 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient397)
  (cocktail-part2 cocktail1 ingredient67)
)
 (:goal
  (and
      (contains shot119 cocktail1)
)))
