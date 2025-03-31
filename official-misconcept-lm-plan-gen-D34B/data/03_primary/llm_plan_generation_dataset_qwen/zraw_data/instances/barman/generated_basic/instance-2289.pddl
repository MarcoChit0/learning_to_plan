(define (problem prob)
 (:domain barman)
 (:objects 
      shaker296 - shaker
      left right - hand
      shot419 - shot
      ingredient200 ingredient495 ingredient5 - ingredient
      cocktail270 - cocktail
      dispenser298 dispenser135 dispenser206 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker296)
  (ontable shot419)
  (dispenses dispenser298 ingredient200)
  (dispenses dispenser135 ingredient495)
  (dispenses dispenser206 ingredient5)
  (clean shaker296)
  (clean shot419)
  (empty shaker296)
  (empty shot419)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker296 l0)
  (shaker-level shaker296 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail270 ingredient200)
  (cocktail-part2 cocktail270 ingredient5)
)
 (:goal
  (and
      (contains shot419 cocktail270)
)))
