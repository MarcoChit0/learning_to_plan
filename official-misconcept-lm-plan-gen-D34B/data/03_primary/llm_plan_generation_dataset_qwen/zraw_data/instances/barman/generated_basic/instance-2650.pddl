(define (problem prob)
 (:domain barman)
 (:objects 
      shaker377 - shaker
      left right - hand
      shot3 shot216 - shot
      ingredient91 ingredient490 ingredient11 - ingredient
      cocktail386 - cocktail
      dispenser190 dispenser15 dispenser440 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker377)
  (ontable shot3)
  (ontable shot216)
  (dispenses dispenser190 ingredient91)
  (dispenses dispenser15 ingredient490)
  (dispenses dispenser440 ingredient11)
  (clean shaker377)
  (clean shot3)
  (clean shot216)
  (empty shaker377)
  (empty shot3)
  (empty shot216)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker377 l0)
  (shaker-level shaker377 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail386 ingredient490)
  (cocktail-part2 cocktail386 ingredient91)
)
 (:goal
  (and
      (contains shot3 cocktail386)
)))
