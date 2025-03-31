(define (problem prob)
 (:domain barman)
 (:objects 
      shaker106 - shaker
      left right - hand
      shot185 shot21 shot216 - shot
      ingredient440 ingredient113 ingredient417 ingredient432 - ingredient
      cocktail309 - cocktail
      dispenser329 dispenser214 dispenser495 dispenser110 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker106)
  (ontable shot185)
  (ontable shot21)
  (ontable shot216)
  (dispenses dispenser329 ingredient440)
  (dispenses dispenser214 ingredient113)
  (dispenses dispenser495 ingredient417)
  (dispenses dispenser110 ingredient432)
  (clean shaker106)
  (clean shot185)
  (clean shot21)
  (clean shot216)
  (empty shaker106)
  (empty shot185)
  (empty shot21)
  (empty shot216)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker106 l0)
  (shaker-level shaker106 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail309 ingredient440)
  (cocktail-part2 cocktail309 ingredient417)
)
 (:goal
  (and
      (contains shot185 cocktail309)
      (contains shot21 cocktail309)
)))
