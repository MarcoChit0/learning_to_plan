(define (problem prob)
 (:domain barman)
 (:objects 
      shaker369 - shaker
      left right - hand
      shot495 shot419 - shot
      ingredient269 ingredient230 ingredient259 ingredient224 - ingredient
      cocktail348 - cocktail
      dispenser365 dispenser198 dispenser449 dispenser482 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker369)
  (ontable shot495)
  (ontable shot419)
  (dispenses dispenser365 ingredient269)
  (dispenses dispenser198 ingredient230)
  (dispenses dispenser449 ingredient259)
  (dispenses dispenser482 ingredient224)
  (clean shaker369)
  (clean shot495)
  (clean shot419)
  (empty shaker369)
  (empty shot495)
  (empty shot419)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker369 l0)
  (shaker-level shaker369 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail348 ingredient259)
  (cocktail-part2 cocktail348 ingredient230)
)
 (:goal
  (and
      (contains shot495 cocktail348)
)))
