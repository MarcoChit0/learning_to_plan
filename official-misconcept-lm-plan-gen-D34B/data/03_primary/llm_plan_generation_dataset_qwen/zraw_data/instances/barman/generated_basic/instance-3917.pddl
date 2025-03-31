(define (problem prob)
 (:domain barman)
 (:objects 
      shaker236 - shaker
      left right - hand
      shot370 - shot
      ingredient25 ingredient323 ingredient370 ingredient0 - ingredient
      cocktail362 - cocktail
      dispenser461 dispenser13 dispenser34 dispenser49 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker236)
  (ontable shot370)
  (dispenses dispenser461 ingredient25)
  (dispenses dispenser13 ingredient323)
  (dispenses dispenser34 ingredient370)
  (dispenses dispenser49 ingredient0)
  (clean shaker236)
  (clean shot370)
  (empty shaker236)
  (empty shot370)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker236 l0)
  (shaker-level shaker236 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail362 ingredient323)
  (cocktail-part2 cocktail362 ingredient370)
)
 (:goal
  (and
      (contains shot370 cocktail362)
)))
