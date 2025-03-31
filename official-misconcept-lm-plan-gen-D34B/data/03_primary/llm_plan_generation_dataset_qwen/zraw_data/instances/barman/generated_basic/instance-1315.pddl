(define (problem prob)
 (:domain barman)
 (:objects 
      shaker208 - shaker
      left right - hand
      shot235 - shot
      ingredient473 ingredient303 ingredient72 ingredient104 - ingredient
      cocktail248 - cocktail
      dispenser413 dispenser331 dispenser404 dispenser454 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker208)
  (ontable shot235)
  (dispenses dispenser413 ingredient473)
  (dispenses dispenser331 ingredient303)
  (dispenses dispenser404 ingredient72)
  (dispenses dispenser454 ingredient104)
  (clean shaker208)
  (clean shot235)
  (empty shaker208)
  (empty shot235)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker208 l0)
  (shaker-level shaker208 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail248 ingredient72)
  (cocktail-part2 cocktail248 ingredient303)
)
 (:goal
  (and
      (contains shot235 cocktail248)
)))
