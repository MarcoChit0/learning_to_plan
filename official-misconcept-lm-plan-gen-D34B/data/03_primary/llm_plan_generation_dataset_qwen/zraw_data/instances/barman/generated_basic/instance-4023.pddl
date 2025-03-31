(define (problem prob)
 (:domain barman)
 (:objects 
      shaker369 - shaker
      left right - hand
      shot99 shot194 - shot
      ingredient484 ingredient281 ingredient418 - ingredient
      cocktail117 - cocktail
      dispenser463 dispenser140 dispenser405 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker369)
  (ontable shot99)
  (ontable shot194)
  (dispenses dispenser463 ingredient484)
  (dispenses dispenser140 ingredient281)
  (dispenses dispenser405 ingredient418)
  (clean shaker369)
  (clean shot99)
  (clean shot194)
  (empty shaker369)
  (empty shot99)
  (empty shot194)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker369 l0)
  (shaker-level shaker369 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail117 ingredient484)
  (cocktail-part2 cocktail117 ingredient281)
)
 (:goal
  (and
      (contains shot99 cocktail117)
)))
