(define (problem prob)
 (:domain barman)
 (:objects 
      shaker33 - shaker
      left right - hand
      shot38 - shot
      ingredient391 ingredient171 ingredient304 ingredient86 - ingredient
      cocktail125 - cocktail
      dispenser390 dispenser490 dispenser145 dispenser73 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker33)
  (ontable shot38)
  (dispenses dispenser390 ingredient391)
  (dispenses dispenser490 ingredient171)
  (dispenses dispenser145 ingredient304)
  (dispenses dispenser73 ingredient86)
  (clean shaker33)
  (clean shot38)
  (empty shaker33)
  (empty shot38)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker33 l0)
  (shaker-level shaker33 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail125 ingredient171)
  (cocktail-part2 cocktail125 ingredient86)
)
 (:goal
  (and
      (contains shot38 cocktail125)
)))
