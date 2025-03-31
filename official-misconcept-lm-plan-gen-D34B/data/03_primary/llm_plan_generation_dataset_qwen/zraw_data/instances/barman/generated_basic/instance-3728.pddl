(define (problem prob)
 (:domain barman)
 (:objects 
      shaker283 - shaker
      left right - hand
      shot22 - shot
      ingredient37 ingredient197 ingredient121 ingredient55 - ingredient
      cocktail462 - cocktail
      dispenser23 dispenser343 dispenser321 dispenser434 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker283)
  (ontable shot22)
  (dispenses dispenser23 ingredient37)
  (dispenses dispenser343 ingredient197)
  (dispenses dispenser321 ingredient121)
  (dispenses dispenser434 ingredient55)
  (clean shaker283)
  (clean shot22)
  (empty shaker283)
  (empty shot22)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker283 l0)
  (shaker-level shaker283 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail462 ingredient55)
  (cocktail-part2 cocktail462 ingredient121)
)
 (:goal
  (and
      (contains shot22 cocktail462)
)))
