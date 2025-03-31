(define (problem prob)
 (:domain barman)
 (:objects 
      shaker405 - shaker
      left right - hand
      shot380 - shot
      ingredient234 ingredient280 - ingredient
      cocktail143 - cocktail
      dispenser366 dispenser349 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker405)
  (ontable shot380)
  (dispenses dispenser366 ingredient234)
  (dispenses dispenser349 ingredient280)
  (clean shaker405)
  (clean shot380)
  (empty shaker405)
  (empty shot380)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker405 l0)
  (shaker-level shaker405 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail143 ingredient280)
  (cocktail-part2 cocktail143 ingredient234)
)
 (:goal
  (and
      (contains shot380 cocktail143)
)))
