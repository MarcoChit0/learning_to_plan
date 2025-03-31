(define (problem prob)
 (:domain barman)
 (:objects 
      shaker497 - shaker
      left right - hand
      shot70 shot453 shot491 - shot
      ingredient208 ingredient417 ingredient475 - ingredient
      cocktail423 - cocktail
      dispenser452 dispenser392 dispenser426 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker497)
  (ontable shot70)
  (ontable shot453)
  (ontable shot491)
  (dispenses dispenser452 ingredient208)
  (dispenses dispenser392 ingredient417)
  (dispenses dispenser426 ingredient475)
  (clean shaker497)
  (clean shot70)
  (clean shot453)
  (clean shot491)
  (empty shaker497)
  (empty shot70)
  (empty shot453)
  (empty shot491)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker497 l0)
  (shaker-level shaker497 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail423 ingredient417)
  (cocktail-part2 cocktail423 ingredient475)
)
 (:goal
  (and
      (contains shot70 cocktail423)
      (contains shot453 ingredient417)
)))
