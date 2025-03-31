(define (problem prob)
 (:domain barman)
 (:objects 
      shaker36 - shaker
      left right - hand
      shot457 shot382 - shot
      ingredient300 ingredient315 ingredient446 - ingredient
      cocktail88 - cocktail
      dispenser123 dispenser490 dispenser162 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker36)
  (ontable shot457)
  (ontable shot382)
  (dispenses dispenser123 ingredient300)
  (dispenses dispenser490 ingredient315)
  (dispenses dispenser162 ingredient446)
  (clean shaker36)
  (clean shot457)
  (clean shot382)
  (empty shaker36)
  (empty shot457)
  (empty shot382)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker36 l0)
  (shaker-level shaker36 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail88 ingredient315)
  (cocktail-part2 cocktail88 ingredient300)
)
 (:goal
  (and
      (contains shot457 cocktail88)
)))
