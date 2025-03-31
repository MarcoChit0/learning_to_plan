(define (problem prob)
 (:domain barman)
 (:objects 
      shaker96 - shaker
      left right - hand
      shot229 - shot
      ingredient277 ingredient415 ingredient356 - ingredient
      cocktail1 - cocktail
      dispenser428 dispenser315 dispenser475 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker96)
  (ontable shot229)
  (dispenses dispenser428 ingredient277)
  (dispenses dispenser315 ingredient415)
  (dispenses dispenser475 ingredient356)
  (clean shaker96)
  (clean shot229)
  (empty shaker96)
  (empty shot229)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker96 l0)
  (shaker-level shaker96 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient415)
  (cocktail-part2 cocktail1 ingredient356)
)
 (:goal
  (and
      (contains shot229 cocktail1)
)))
