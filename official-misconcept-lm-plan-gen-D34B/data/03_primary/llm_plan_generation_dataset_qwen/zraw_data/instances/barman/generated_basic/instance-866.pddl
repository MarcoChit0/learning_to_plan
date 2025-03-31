(define (problem prob)
 (:domain barman)
 (:objects 
      shaker303 - shaker
      left right - hand
      shot90 shot71 shot43 - shot
      ingredient5 ingredient58 ingredient214 ingredient470 - ingredient
      cocktail1 - cocktail
      dispenser145 dispenser437 dispenser192 dispenser298 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker303)
  (ontable shot90)
  (ontable shot71)
  (ontable shot43)
  (dispenses dispenser145 ingredient5)
  (dispenses dispenser437 ingredient58)
  (dispenses dispenser192 ingredient214)
  (dispenses dispenser298 ingredient470)
  (clean shaker303)
  (clean shot90)
  (clean shot71)
  (clean shot43)
  (empty shaker303)
  (empty shot90)
  (empty shot71)
  (empty shot43)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker303 l0)
  (shaker-level shaker303 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient58)
  (cocktail-part2 cocktail1 ingredient214)
)
 (:goal
  (and
      (contains shot90 cocktail1)
      (contains shot71 cocktail1)
)))
