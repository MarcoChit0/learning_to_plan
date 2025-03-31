(define (problem prob)
 (:domain barman)
 (:objects 
      shaker409 - shaker
      left right - hand
      shot273 shot48 - shot
      ingredient98 ingredient159 ingredient352 ingredient350 - ingredient
      cocktail309 - cocktail
      dispenser487 dispenser167 dispenser35 dispenser262 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker409)
  (ontable shot273)
  (ontable shot48)
  (dispenses dispenser487 ingredient98)
  (dispenses dispenser167 ingredient159)
  (dispenses dispenser35 ingredient352)
  (dispenses dispenser262 ingredient350)
  (clean shaker409)
  (clean shot273)
  (clean shot48)
  (empty shaker409)
  (empty shot273)
  (empty shot48)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker409 l0)
  (shaker-level shaker409 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail309 ingredient159)
  (cocktail-part2 cocktail309 ingredient98)
)
 (:goal
  (and
      (contains shot273 cocktail309)
)))
