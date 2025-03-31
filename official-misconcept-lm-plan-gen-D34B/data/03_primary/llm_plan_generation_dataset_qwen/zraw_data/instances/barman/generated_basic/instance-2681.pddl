(define (problem prob)
 (:domain barman)
 (:objects 
      shaker362 - shaker
      left right - hand
      shot234 shot482 - shot
      ingredient230 ingredient385 ingredient434 - ingredient
      cocktail197 - cocktail
      dispenser104 dispenser379 dispenser175 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker362)
  (ontable shot234)
  (ontable shot482)
  (dispenses dispenser104 ingredient230)
  (dispenses dispenser379 ingredient385)
  (dispenses dispenser175 ingredient434)
  (clean shaker362)
  (clean shot234)
  (clean shot482)
  (empty shaker362)
  (empty shot234)
  (empty shot482)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker362 l0)
  (shaker-level shaker362 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail197 ingredient385)
  (cocktail-part2 cocktail197 ingredient230)
)
 (:goal
  (and
      (contains shot234 cocktail197)
)))
