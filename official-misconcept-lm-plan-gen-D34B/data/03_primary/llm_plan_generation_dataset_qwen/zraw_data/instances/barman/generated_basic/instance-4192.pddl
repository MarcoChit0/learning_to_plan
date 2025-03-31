(define (problem prob)
 (:domain barman)
 (:objects 
      shaker96 - shaker
      left right - hand
      shot35 - shot
      ingredient118 ingredient269 ingredient271 ingredient493 - ingredient
      cocktail83 - cocktail
      dispenser79 dispenser225 dispenser320 dispenser49 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker96)
  (ontable shot35)
  (dispenses dispenser79 ingredient118)
  (dispenses dispenser225 ingredient269)
  (dispenses dispenser320 ingredient271)
  (dispenses dispenser49 ingredient493)
  (clean shaker96)
  (clean shot35)
  (empty shaker96)
  (empty shot35)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker96 l0)
  (shaker-level shaker96 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail83 ingredient269)
  (cocktail-part2 cocktail83 ingredient118)
)
 (:goal
  (and
      (contains shot35 cocktail83)
)))
