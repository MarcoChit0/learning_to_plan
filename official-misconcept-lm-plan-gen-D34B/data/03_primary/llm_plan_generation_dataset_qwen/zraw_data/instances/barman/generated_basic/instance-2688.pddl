(define (problem prob)
 (:domain barman)
 (:objects 
      shaker188 - shaker
      left right - hand
      shot236 shot312 - shot
      ingredient446 ingredient233 ingredient433 - ingredient
      cocktail53 - cocktail
      dispenser208 dispenser140 dispenser229 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker188)
  (ontable shot236)
  (ontable shot312)
  (dispenses dispenser208 ingredient446)
  (dispenses dispenser140 ingredient233)
  (dispenses dispenser229 ingredient433)
  (clean shaker188)
  (clean shot236)
  (clean shot312)
  (empty shaker188)
  (empty shot236)
  (empty shot312)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker188 l0)
  (shaker-level shaker188 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail53 ingredient446)
  (cocktail-part2 cocktail53 ingredient233)
)
 (:goal
  (and
      (contains shot236 cocktail53)
)))
