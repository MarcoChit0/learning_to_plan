(define (problem prob)
 (:domain barman)
 (:objects 
      shaker383 - shaker
      left right - hand
      shot330 - shot
      ingredient360 ingredient178 - ingredient
      cocktail441 - cocktail
      dispenser125 dispenser349 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker383)
  (ontable shot330)
  (dispenses dispenser125 ingredient360)
  (dispenses dispenser349 ingredient178)
  (clean shaker383)
  (clean shot330)
  (empty shaker383)
  (empty shot330)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker383 l0)
  (shaker-level shaker383 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail441 ingredient360)
  (cocktail-part2 cocktail441 ingredient178)
)
 (:goal
  (and
      (contains shot330 cocktail441)
)))
