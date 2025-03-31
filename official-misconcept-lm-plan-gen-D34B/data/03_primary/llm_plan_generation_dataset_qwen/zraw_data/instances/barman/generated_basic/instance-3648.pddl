(define (problem prob)
 (:domain barman)
 (:objects 
      shaker304 - shaker
      left right - hand
      shot90 shot47 - shot
      ingredient465 ingredient267 ingredient63 ingredient412 - ingredient
      cocktail253 - cocktail
      dispenser283 dispenser334 dispenser372 dispenser479 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker304)
  (ontable shot90)
  (ontable shot47)
  (dispenses dispenser283 ingredient465)
  (dispenses dispenser334 ingredient267)
  (dispenses dispenser372 ingredient63)
  (dispenses dispenser479 ingredient412)
  (clean shaker304)
  (clean shot90)
  (clean shot47)
  (empty shaker304)
  (empty shot90)
  (empty shot47)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker304 l0)
  (shaker-level shaker304 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail253 ingredient465)
  (cocktail-part2 cocktail253 ingredient267)
)
 (:goal
  (and
      (contains shot90 cocktail253)
)))
