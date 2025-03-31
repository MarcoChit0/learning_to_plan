(define (problem prob)
 (:domain barman)
 (:objects 
      shaker262 - shaker
      left right - hand
      shot222 - shot
      ingredient392 ingredient53 - ingredient
      cocktail331 - cocktail
      dispenser19 dispenser217 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker262)
  (ontable shot222)
  (dispenses dispenser19 ingredient392)
  (dispenses dispenser217 ingredient53)
  (clean shaker262)
  (clean shot222)
  (empty shaker262)
  (empty shot222)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker262 l0)
  (shaker-level shaker262 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail331 ingredient53)
  (cocktail-part2 cocktail331 ingredient392)
)
 (:goal
  (and
      (contains shot222 cocktail331)
)))
