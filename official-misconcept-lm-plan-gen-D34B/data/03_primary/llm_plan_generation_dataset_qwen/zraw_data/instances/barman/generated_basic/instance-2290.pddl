(define (problem prob)
 (:domain barman)
 (:objects 
      shaker101 - shaker
      left right - hand
      shot112 - shot
      ingredient389 ingredient471 ingredient34 - ingredient
      cocktail104 - cocktail
      dispenser292 dispenser410 dispenser31 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker101)
  (ontable shot112)
  (dispenses dispenser292 ingredient389)
  (dispenses dispenser410 ingredient471)
  (dispenses dispenser31 ingredient34)
  (clean shaker101)
  (clean shot112)
  (empty shaker101)
  (empty shot112)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker101 l0)
  (shaker-level shaker101 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail104 ingredient34)
  (cocktail-part2 cocktail104 ingredient389)
)
 (:goal
  (and
      (contains shot112 cocktail104)
)))
