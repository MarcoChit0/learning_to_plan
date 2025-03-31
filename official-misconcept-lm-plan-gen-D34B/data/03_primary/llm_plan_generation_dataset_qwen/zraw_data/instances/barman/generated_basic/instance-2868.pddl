(define (problem prob)
 (:domain barman)
 (:objects 
      shaker382 - shaker
      left right - hand
      shot482 shot496 - shot
      ingredient62 ingredient380 ingredient232 ingredient132 - ingredient
      cocktail489 - cocktail
      dispenser191 dispenser69 dispenser148 dispenser269 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker382)
  (ontable shot482)
  (ontable shot496)
  (dispenses dispenser191 ingredient62)
  (dispenses dispenser69 ingredient380)
  (dispenses dispenser148 ingredient232)
  (dispenses dispenser269 ingredient132)
  (clean shaker382)
  (clean shot482)
  (clean shot496)
  (empty shaker382)
  (empty shot482)
  (empty shot496)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker382 l0)
  (shaker-level shaker382 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail489 ingredient232)
  (cocktail-part2 cocktail489 ingredient62)
)
 (:goal
  (and
      (contains shot482 cocktail489)
)))
