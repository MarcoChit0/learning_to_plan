(define (problem prob)
 (:domain barman)
 (:objects 
      shaker112 - shaker
      left right - hand
      shot215 - shot
      ingredient80 ingredient416 ingredient67 ingredient446 - ingredient
      cocktail1 - cocktail
      dispenser199 dispenser452 dispenser195 dispenser380 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker112)
  (ontable shot215)
  (dispenses dispenser199 ingredient80)
  (dispenses dispenser452 ingredient416)
  (dispenses dispenser195 ingredient67)
  (dispenses dispenser380 ingredient446)
  (clean shaker112)
  (clean shot215)
  (empty shaker112)
  (empty shot215)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker112 l0)
  (shaker-level shaker112 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient67)
  (cocktail-part2 cocktail1 ingredient416)
)
 (:goal
  (and
      (contains shot215 cocktail1)
)))
