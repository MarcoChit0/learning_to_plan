(define (problem prob)
 (:domain barman)
 (:objects 
      shaker419 - shaker
      left right - hand
      shot499 shot458 - shot
      ingredient260 ingredient256 ingredient465 - ingredient
      cocktail174 - cocktail
      dispenser495 dispenser143 dispenser120 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker419)
  (ontable shot499)
  (ontable shot458)
  (dispenses dispenser495 ingredient260)
  (dispenses dispenser143 ingredient256)
  (dispenses dispenser120 ingredient465)
  (clean shaker419)
  (clean shot499)
  (clean shot458)
  (empty shaker419)
  (empty shot499)
  (empty shot458)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker419 l0)
  (shaker-level shaker419 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail174 ingredient256)
  (cocktail-part2 cocktail174 ingredient260)
)
 (:goal
  (and
      (contains shot499 cocktail174)
)))
