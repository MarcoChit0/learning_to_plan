(define (problem prob)
 (:domain barman)
 (:objects 
      shaker155 - shaker
      left right - hand
      shot69 - shot
      ingredient249 ingredient319 ingredient182 - ingredient
      cocktail1 - cocktail
      dispenser419 dispenser98 dispenser85 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker155)
  (ontable shot69)
  (dispenses dispenser419 ingredient249)
  (dispenses dispenser98 ingredient319)
  (dispenses dispenser85 ingredient182)
  (clean shaker155)
  (clean shot69)
  (empty shaker155)
  (empty shot69)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker155 l0)
  (shaker-level shaker155 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient249)
  (cocktail-part2 cocktail1 ingredient182)
)
 (:goal
  (and
      (contains shot69 cocktail1)
)))
