(define (problem prob)
 (:domain barman)
 (:objects 
      shaker1 - shaker
      left right - hand
      shot469 shot90 - shot
      ingredient385 ingredient180 ingredient473 - ingredient
      cocktail403 - cocktail
      dispenser201 dispenser347 dispenser236 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker1)
  (ontable shot469)
  (ontable shot90)
  (dispenses dispenser201 ingredient385)
  (dispenses dispenser347 ingredient180)
  (dispenses dispenser236 ingredient473)
  (clean shaker1)
  (clean shot469)
  (clean shot90)
  (empty shaker1)
  (empty shot469)
  (empty shot90)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker1 l0)
  (shaker-level shaker1 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail403 ingredient385)
  (cocktail-part2 cocktail403 ingredient473)
)
 (:goal
  (and
      (contains shot469 cocktail403)
)))
