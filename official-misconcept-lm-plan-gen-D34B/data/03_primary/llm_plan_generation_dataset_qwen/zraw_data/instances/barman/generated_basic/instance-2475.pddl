(define (problem prob)
 (:domain barman)
 (:objects 
      shaker233 - shaker
      left right - hand
      shot177 - shot
      ingredient384 ingredient419 ingredient242 - ingredient
      cocktail314 - cocktail
      dispenser47 dispenser408 dispenser34 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker233)
  (ontable shot177)
  (dispenses dispenser47 ingredient384)
  (dispenses dispenser408 ingredient419)
  (dispenses dispenser34 ingredient242)
  (clean shaker233)
  (clean shot177)
  (empty shaker233)
  (empty shot177)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker233 l0)
  (shaker-level shaker233 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail314 ingredient384)
  (cocktail-part2 cocktail314 ingredient419)
)
 (:goal
  (and
      (contains shot177 cocktail314)
)))
