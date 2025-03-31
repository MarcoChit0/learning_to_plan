(define (problem prob)
 (:domain barman)
 (:objects 
      shaker51 - shaker
      left right - hand
      shot484 - shot
      ingredient180 ingredient320 - ingredient
      cocktail1 - cocktail
      dispenser56 dispenser130 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker51)
  (ontable shot484)
  (dispenses dispenser56 ingredient180)
  (dispenses dispenser130 ingredient320)
  (clean shaker51)
  (clean shot484)
  (empty shaker51)
  (empty shot484)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker51 l0)
  (shaker-level shaker51 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient320)
  (cocktail-part2 cocktail1 ingredient180)
)
 (:goal
  (and
      (contains shot484 cocktail1)
)))
