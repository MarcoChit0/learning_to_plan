(define (problem prob)
 (:domain barman)
 (:objects 
      shaker314 - shaker
      left right - hand
      shot253 - shot
      ingredient233 ingredient110 ingredient272 - ingredient
      cocktail77 - cocktail
      dispenser275 dispenser404 dispenser123 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker314)
  (ontable shot253)
  (dispenses dispenser275 ingredient233)
  (dispenses dispenser404 ingredient110)
  (dispenses dispenser123 ingredient272)
  (clean shaker314)
  (clean shot253)
  (empty shaker314)
  (empty shot253)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker314 l0)
  (shaker-level shaker314 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail77 ingredient272)
  (cocktail-part2 cocktail77 ingredient233)
)
 (:goal
  (and
      (contains shot253 cocktail77)
)))
