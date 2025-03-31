(define (problem prob)
 (:domain barman)
 (:objects 
      shaker6 - shaker
      left right - hand
      shot83 shot160 - shot
      ingredient324 ingredient214 ingredient310 - ingredient
      cocktail468 - cocktail
      dispenser262 dispenser150 dispenser319 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker6)
  (ontable shot83)
  (ontable shot160)
  (dispenses dispenser262 ingredient324)
  (dispenses dispenser150 ingredient214)
  (dispenses dispenser319 ingredient310)
  (clean shaker6)
  (clean shot83)
  (clean shot160)
  (empty shaker6)
  (empty shot83)
  (empty shot160)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker6 l0)
  (shaker-level shaker6 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail468 ingredient214)
  (cocktail-part2 cocktail468 ingredient324)
)
 (:goal
  (and
      (contains shot83 cocktail468)
)))
