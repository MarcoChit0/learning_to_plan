(define (problem prob)
 (:domain barman)
 (:objects 
      shaker420 - shaker
      left right - hand
      shot71 - shot
      ingredient320 ingredient256 ingredient214 ingredient92 - ingredient
      cocktail10 - cocktail
      dispenser243 dispenser25 dispenser292 dispenser449 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker420)
  (ontable shot71)
  (dispenses dispenser243 ingredient320)
  (dispenses dispenser25 ingredient256)
  (dispenses dispenser292 ingredient214)
  (dispenses dispenser449 ingredient92)
  (clean shaker420)
  (clean shot71)
  (empty shaker420)
  (empty shot71)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker420 l0)
  (shaker-level shaker420 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail10 ingredient214)
  (cocktail-part2 cocktail10 ingredient320)
)
 (:goal
  (and
      (contains shot71 cocktail10)
)))
