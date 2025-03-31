(define (problem prob)
 (:domain barman)
 (:objects 
      shaker364 - shaker
      left right - hand
      shot194 shot164 shot451 - shot
      ingredient130 ingredient490 ingredient150 ingredient484 - ingredient
      cocktail412 - cocktail
      dispenser426 dispenser360 dispenser263 dispenser113 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker364)
  (ontable shot194)
  (ontable shot164)
  (ontable shot451)
  (dispenses dispenser426 ingredient130)
  (dispenses dispenser360 ingredient490)
  (dispenses dispenser263 ingredient150)
  (dispenses dispenser113 ingredient484)
  (clean shaker364)
  (clean shot194)
  (clean shot164)
  (clean shot451)
  (empty shaker364)
  (empty shot194)
  (empty shot164)
  (empty shot451)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker364 l0)
  (shaker-level shaker364 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail412 ingredient490)
  (cocktail-part2 cocktail412 ingredient484)
)
 (:goal
  (and
      (contains shot194 cocktail412)
      (contains shot164 cocktail412)
)))
