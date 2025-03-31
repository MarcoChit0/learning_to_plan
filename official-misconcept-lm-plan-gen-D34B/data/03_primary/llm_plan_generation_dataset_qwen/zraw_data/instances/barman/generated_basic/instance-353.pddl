(define (problem prob)
 (:domain barman)
 (:objects 
      shaker291 - shaker
      left right - hand
      shot84 shot258 - shot
      ingredient419 ingredient211 ingredient96 ingredient116 - ingredient
      cocktail1 - cocktail
      dispenser90 dispenser34 dispenser214 dispenser210 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker291)
  (ontable shot84)
  (ontable shot258)
  (dispenses dispenser90 ingredient419)
  (dispenses dispenser34 ingredient211)
  (dispenses dispenser214 ingredient96)
  (dispenses dispenser210 ingredient116)
  (clean shaker291)
  (clean shot84)
  (clean shot258)
  (empty shaker291)
  (empty shot84)
  (empty shot258)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker291 l0)
  (shaker-level shaker291 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient96)
  (cocktail-part2 cocktail1 ingredient116)
)
 (:goal
  (and
      (contains shot84 cocktail1)
)))
