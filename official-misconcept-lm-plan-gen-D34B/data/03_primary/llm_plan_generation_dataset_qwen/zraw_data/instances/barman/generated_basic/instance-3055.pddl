(define (problem prob)
 (:domain barman)
 (:objects 
      shaker454 - shaker
      left right - hand
      shot457 shot22 - shot
      ingredient373 ingredient55 ingredient283 - ingredient
      cocktail249 - cocktail
      dispenser247 dispenser419 dispenser63 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker454)
  (ontable shot457)
  (ontable shot22)
  (dispenses dispenser247 ingredient373)
  (dispenses dispenser419 ingredient55)
  (dispenses dispenser63 ingredient283)
  (clean shaker454)
  (clean shot457)
  (clean shot22)
  (empty shaker454)
  (empty shot457)
  (empty shot22)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker454 l0)
  (shaker-level shaker454 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail249 ingredient55)
  (cocktail-part2 cocktail249 ingredient373)
)
 (:goal
  (and
      (contains shot457 cocktail249)
)))
