(define (problem prob)
 (:domain barman)
 (:objects 
      shaker162 - shaker
      left right - hand
      shot79 - shot
      ingredient223 ingredient452 ingredient330 - ingredient
      cocktail359 - cocktail
      dispenser5 dispenser436 dispenser40 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker162)
  (ontable shot79)
  (dispenses dispenser5 ingredient223)
  (dispenses dispenser436 ingredient452)
  (dispenses dispenser40 ingredient330)
  (clean shaker162)
  (clean shot79)
  (empty shaker162)
  (empty shot79)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker162 l0)
  (shaker-level shaker162 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail359 ingredient452)
  (cocktail-part2 cocktail359 ingredient330)
)
 (:goal
  (and
      (contains shot79 cocktail359)
)))
