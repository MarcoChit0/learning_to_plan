(define (problem prob)
 (:domain barman)
 (:objects 
      shaker293 - shaker
      left right - hand
      shot242 - shot
      ingredient24 ingredient219 ingredient325 ingredient71 - ingredient
      cocktail35 - cocktail
      dispenser328 dispenser81 dispenser17 dispenser145 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker293)
  (ontable shot242)
  (dispenses dispenser328 ingredient24)
  (dispenses dispenser81 ingredient219)
  (dispenses dispenser17 ingredient325)
  (dispenses dispenser145 ingredient71)
  (clean shaker293)
  (clean shot242)
  (empty shaker293)
  (empty shot242)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker293 l0)
  (shaker-level shaker293 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail35 ingredient325)
  (cocktail-part2 cocktail35 ingredient219)
)
 (:goal
  (and
      (contains shot242 cocktail35)
)))
