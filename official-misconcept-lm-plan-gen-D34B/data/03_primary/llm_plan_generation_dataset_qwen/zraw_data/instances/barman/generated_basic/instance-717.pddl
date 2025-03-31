(define (problem prob)
 (:domain barman)
 (:objects 
      shaker262 - shaker
      left right - hand
      shot249 - shot
      ingredient65 ingredient243 ingredient175 - ingredient
      cocktail1 - cocktail
      dispenser431 dispenser367 dispenser374 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker262)
  (ontable shot249)
  (dispenses dispenser431 ingredient65)
  (dispenses dispenser367 ingredient243)
  (dispenses dispenser374 ingredient175)
  (clean shaker262)
  (clean shot249)
  (empty shaker262)
  (empty shot249)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker262 l0)
  (shaker-level shaker262 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient243)
  (cocktail-part2 cocktail1 ingredient65)
)
 (:goal
  (and
      (contains shot249 cocktail1)
)))
