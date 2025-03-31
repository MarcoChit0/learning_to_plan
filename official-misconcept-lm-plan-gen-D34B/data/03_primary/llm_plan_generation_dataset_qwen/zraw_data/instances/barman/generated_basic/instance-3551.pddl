(define (problem prob)
 (:domain barman)
 (:objects 
      shaker465 - shaker
      left right - hand
      shot204 - shot
      ingredient339 ingredient304 ingredient382 - ingredient
      cocktail1 - cocktail
      dispenser201 dispenser219 dispenser5 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker465)
  (ontable shot204)
  (dispenses dispenser201 ingredient339)
  (dispenses dispenser219 ingredient304)
  (dispenses dispenser5 ingredient382)
  (clean shaker465)
  (clean shot204)
  (empty shaker465)
  (empty shot204)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker465 l0)
  (shaker-level shaker465 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient339)
  (cocktail-part2 cocktail1 ingredient382)
)
 (:goal
  (and
      (contains shot204 cocktail1)
)))
