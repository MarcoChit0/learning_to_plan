(define (problem prob)
 (:domain barman)
 (:objects 
      shaker117 - shaker
      left right - hand
      shot215 shot6 - shot
      ingredient209 ingredient243 ingredient349 ingredient11 - ingredient
      cocktail461 - cocktail
      dispenser271 dispenser13 dispenser286 dispenser374 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker117)
  (ontable shot215)
  (ontable shot6)
  (dispenses dispenser271 ingredient209)
  (dispenses dispenser13 ingredient243)
  (dispenses dispenser286 ingredient349)
  (dispenses dispenser374 ingredient11)
  (clean shaker117)
  (clean shot215)
  (clean shot6)
  (empty shaker117)
  (empty shot215)
  (empty shot6)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker117 l0)
  (shaker-level shaker117 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail461 ingredient349)
  (cocktail-part2 cocktail461 ingredient243)
)
 (:goal
  (and
      (contains shot215 cocktail461)
)))
