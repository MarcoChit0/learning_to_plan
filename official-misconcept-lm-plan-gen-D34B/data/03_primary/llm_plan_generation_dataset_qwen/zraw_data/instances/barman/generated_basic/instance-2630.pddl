(define (problem prob)
 (:domain barman)
 (:objects 
      shaker332 - shaker
      left right - hand
      shot12 shot84 - shot
      ingredient495 ingredient326 ingredient10 - ingredient
      cocktail205 - cocktail
      dispenser410 dispenser336 dispenser320 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker332)
  (ontable shot12)
  (ontable shot84)
  (dispenses dispenser410 ingredient495)
  (dispenses dispenser336 ingredient326)
  (dispenses dispenser320 ingredient10)
  (clean shaker332)
  (clean shot12)
  (clean shot84)
  (empty shaker332)
  (empty shot12)
  (empty shot84)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker332 l0)
  (shaker-level shaker332 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail205 ingredient495)
  (cocktail-part2 cocktail205 ingredient10)
)
 (:goal
  (and
      (contains shot12 cocktail205)
)))
