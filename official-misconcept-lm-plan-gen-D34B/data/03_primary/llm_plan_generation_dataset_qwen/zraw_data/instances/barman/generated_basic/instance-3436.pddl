(define (problem prob)
 (:domain barman)
 (:objects 
      shaker131 - shaker
      left right - hand
      shot130 shot266 - shot
      ingredient329 ingredient341 ingredient86 ingredient302 - ingredient
      cocktail267 - cocktail
      dispenser140 dispenser34 dispenser162 dispenser390 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker131)
  (ontable shot130)
  (ontable shot266)
  (dispenses dispenser140 ingredient329)
  (dispenses dispenser34 ingredient341)
  (dispenses dispenser162 ingredient86)
  (dispenses dispenser390 ingredient302)
  (clean shaker131)
  (clean shot130)
  (clean shot266)
  (empty shaker131)
  (empty shot130)
  (empty shot266)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker131 l0)
  (shaker-level shaker131 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail267 ingredient329)
  (cocktail-part2 cocktail267 ingredient341)
)
 (:goal
  (and
      (contains shot130 cocktail267)
)))
