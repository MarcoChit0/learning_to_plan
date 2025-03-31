(define (problem prob)
 (:domain barman)
 (:objects 
      shaker212 - shaker
      left right - hand
      shot279 shot37 shot16 - shot
      ingredient315 ingredient347 ingredient254 ingredient325 - ingredient
      cocktail341 - cocktail
      dispenser378 dispenser137 dispenser183 dispenser390 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker212)
  (ontable shot279)
  (ontable shot37)
  (ontable shot16)
  (dispenses dispenser378 ingredient315)
  (dispenses dispenser137 ingredient347)
  (dispenses dispenser183 ingredient254)
  (dispenses dispenser390 ingredient325)
  (clean shaker212)
  (clean shot279)
  (clean shot37)
  (clean shot16)
  (empty shaker212)
  (empty shot279)
  (empty shot37)
  (empty shot16)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker212 l0)
  (shaker-level shaker212 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail341 ingredient325)
  (cocktail-part2 cocktail341 ingredient315)
)
 (:goal
  (and
      (contains shot279 cocktail341)
      (contains shot37 ingredient325)
)))
