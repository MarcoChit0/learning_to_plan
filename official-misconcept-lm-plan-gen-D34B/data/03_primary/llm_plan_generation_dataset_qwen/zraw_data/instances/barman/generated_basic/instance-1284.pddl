(define (problem prob)
 (:domain barman)
 (:objects 
      shaker202 - shaker
      left right - hand
      shot305 - shot
      ingredient207 ingredient473 ingredient282 ingredient416 - ingredient
      cocktail65 - cocktail
      dispenser66 dispenser434 dispenser493 dispenser450 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker202)
  (ontable shot305)
  (dispenses dispenser66 ingredient207)
  (dispenses dispenser434 ingredient473)
  (dispenses dispenser493 ingredient282)
  (dispenses dispenser450 ingredient416)
  (clean shaker202)
  (clean shot305)
  (empty shaker202)
  (empty shot305)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker202 l0)
  (shaker-level shaker202 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail65 ingredient473)
  (cocktail-part2 cocktail65 ingredient207)
)
 (:goal
  (and
      (contains shot305 cocktail65)
)))
