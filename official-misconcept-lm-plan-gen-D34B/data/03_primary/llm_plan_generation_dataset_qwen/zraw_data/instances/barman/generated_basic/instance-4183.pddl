(define (problem prob)
 (:domain barman)
 (:objects 
      shaker297 - shaker
      left right - hand
      shot84 - shot
      ingredient297 ingredient404 ingredient335 ingredient16 - ingredient
      cocktail276 - cocktail
      dispenser478 dispenser216 dispenser349 dispenser334 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker297)
  (ontable shot84)
  (dispenses dispenser478 ingredient297)
  (dispenses dispenser216 ingredient404)
  (dispenses dispenser349 ingredient335)
  (dispenses dispenser334 ingredient16)
  (clean shaker297)
  (clean shot84)
  (empty shaker297)
  (empty shot84)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker297 l0)
  (shaker-level shaker297 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail276 ingredient335)
  (cocktail-part2 cocktail276 ingredient16)
)
 (:goal
  (and
      (contains shot84 cocktail276)
)))
