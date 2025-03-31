(define (problem prob)
 (:domain barman)
 (:objects 
      shaker209 - shaker
      left right - hand
      shot20 shot380 - shot
      ingredient499 ingredient243 ingredient291 ingredient311 - ingredient
      cocktail1 - cocktail
      dispenser458 dispenser440 dispenser399 dispenser425 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker209)
  (ontable shot20)
  (ontable shot380)
  (dispenses dispenser458 ingredient499)
  (dispenses dispenser440 ingredient243)
  (dispenses dispenser399 ingredient291)
  (dispenses dispenser425 ingredient311)
  (clean shaker209)
  (clean shot20)
  (clean shot380)
  (empty shaker209)
  (empty shot20)
  (empty shot380)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker209 l0)
  (shaker-level shaker209 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient499)
  (cocktail-part2 cocktail1 ingredient243)
)
 (:goal
  (and
      (contains shot20 cocktail1)
)))
