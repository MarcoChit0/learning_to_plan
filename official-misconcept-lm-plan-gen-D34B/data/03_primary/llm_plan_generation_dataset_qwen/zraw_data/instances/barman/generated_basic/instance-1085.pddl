(define (problem prob)
 (:domain barman)
 (:objects 
      shaker167 - shaker
      left right - hand
      shot214 - shot
      ingredient99 ingredient182 ingredient416 - ingredient
      cocktail1 - cocktail
      dispenser255 dispenser271 dispenser337 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker167)
  (ontable shot214)
  (dispenses dispenser255 ingredient99)
  (dispenses dispenser271 ingredient182)
  (dispenses dispenser337 ingredient416)
  (clean shaker167)
  (clean shot214)
  (empty shaker167)
  (empty shot214)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker167 l0)
  (shaker-level shaker167 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient99)
  (cocktail-part2 cocktail1 ingredient182)
)
 (:goal
  (and
      (contains shot214 cocktail1)
)))
