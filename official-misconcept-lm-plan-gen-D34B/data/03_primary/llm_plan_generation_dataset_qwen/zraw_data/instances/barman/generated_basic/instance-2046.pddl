(define (problem prob)
 (:domain barman)
 (:objects 
      shaker198 - shaker
      left right - hand
      shot311 - shot
      ingredient397 ingredient200 ingredient19 ingredient167 - ingredient
      cocktail291 - cocktail
      dispenser278 dispenser381 dispenser57 dispenser306 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker198)
  (ontable shot311)
  (dispenses dispenser278 ingredient397)
  (dispenses dispenser381 ingredient200)
  (dispenses dispenser57 ingredient19)
  (dispenses dispenser306 ingredient167)
  (clean shaker198)
  (clean shot311)
  (empty shaker198)
  (empty shot311)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker198 l0)
  (shaker-level shaker198 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail291 ingredient397)
  (cocktail-part2 cocktail291 ingredient200)
)
 (:goal
  (and
      (contains shot311 cocktail291)
)))
