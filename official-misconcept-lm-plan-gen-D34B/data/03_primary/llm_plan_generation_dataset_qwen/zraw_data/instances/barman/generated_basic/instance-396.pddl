(define (problem prob)
 (:domain barman)
 (:objects 
      shaker96 - shaker
      left right - hand
      shot380 shot325 - shot
      ingredient238 ingredient386 - ingredient
      cocktail1 - cocktail
      dispenser143 dispenser397 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker96)
  (ontable shot380)
  (ontable shot325)
  (dispenses dispenser143 ingredient238)
  (dispenses dispenser397 ingredient386)
  (clean shaker96)
  (clean shot380)
  (clean shot325)
  (empty shaker96)
  (empty shot380)
  (empty shot325)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker96 l0)
  (shaker-level shaker96 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient386)
  (cocktail-part2 cocktail1 ingredient238)
)
 (:goal
  (and
      (contains shot380 cocktail1)
)))
