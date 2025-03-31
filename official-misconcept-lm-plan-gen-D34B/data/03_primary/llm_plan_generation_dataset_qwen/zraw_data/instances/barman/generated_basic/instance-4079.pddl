(define (problem prob)
 (:domain barman)
 (:objects 
      shaker461 - shaker
      left right - hand
      shot273 shot71 - shot
      ingredient294 ingredient83 ingredient177 ingredient329 - ingredient
      cocktail348 - cocktail
      dispenser300 dispenser336 dispenser395 dispenser472 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker461)
  (ontable shot273)
  (ontable shot71)
  (dispenses dispenser300 ingredient294)
  (dispenses dispenser336 ingredient83)
  (dispenses dispenser395 ingredient177)
  (dispenses dispenser472 ingredient329)
  (clean shaker461)
  (clean shot273)
  (clean shot71)
  (empty shaker461)
  (empty shot273)
  (empty shot71)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker461 l0)
  (shaker-level shaker461 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail348 ingredient294)
  (cocktail-part2 cocktail348 ingredient83)
)
 (:goal
  (and
      (contains shot273 cocktail348)
)))
