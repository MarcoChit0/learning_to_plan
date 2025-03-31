(define (problem prob)
 (:domain barman)
 (:objects 
      shaker457 - shaker
      left right - hand
      shot471 - shot
      ingredient325 ingredient90 ingredient284 ingredient192 - ingredient
      cocktail1 - cocktail
      dispenser411 dispenser32 dispenser187 dispenser251 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker457)
  (ontable shot471)
  (dispenses dispenser411 ingredient325)
  (dispenses dispenser32 ingredient90)
  (dispenses dispenser187 ingredient284)
  (dispenses dispenser251 ingredient192)
  (clean shaker457)
  (clean shot471)
  (empty shaker457)
  (empty shot471)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker457 l0)
  (shaker-level shaker457 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient192)
  (cocktail-part2 cocktail1 ingredient325)
)
 (:goal
  (and
      (contains shot471 cocktail1)
)))
