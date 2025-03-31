(define (problem prob)
 (:domain barman)
 (:objects 
      shaker309 - shaker
      left right - hand
      shot19 shot211 shot17 - shot
      ingredient344 ingredient428 ingredient419 - ingredient
      cocktail164 - cocktail
      dispenser201 dispenser343 dispenser200 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker309)
  (ontable shot19)
  (ontable shot211)
  (ontable shot17)
  (dispenses dispenser201 ingredient344)
  (dispenses dispenser343 ingredient428)
  (dispenses dispenser200 ingredient419)
  (clean shaker309)
  (clean shot19)
  (clean shot211)
  (clean shot17)
  (empty shaker309)
  (empty shot19)
  (empty shot211)
  (empty shot17)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker309 l0)
  (shaker-level shaker309 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail164 ingredient419)
  (cocktail-part2 cocktail164 ingredient344)
)
 (:goal
  (and
      (contains shot19 cocktail164)
      (contains shot211 cocktail164)
)))
