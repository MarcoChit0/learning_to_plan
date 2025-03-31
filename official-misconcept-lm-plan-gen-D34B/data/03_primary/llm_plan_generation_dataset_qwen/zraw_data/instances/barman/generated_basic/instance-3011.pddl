(define (problem prob)
 (:domain barman)
 (:objects 
      shaker215 - shaker
      left right - hand
      shot446 shot108 - shot
      ingredient119 ingredient280 ingredient382 - ingredient
      cocktail54 - cocktail
      dispenser261 dispenser379 dispenser369 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker215)
  (ontable shot446)
  (ontable shot108)
  (dispenses dispenser261 ingredient119)
  (dispenses dispenser379 ingredient280)
  (dispenses dispenser369 ingredient382)
  (clean shaker215)
  (clean shot446)
  (clean shot108)
  (empty shaker215)
  (empty shot446)
  (empty shot108)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker215 l0)
  (shaker-level shaker215 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail54 ingredient280)
  (cocktail-part2 cocktail54 ingredient119)
)
 (:goal
  (and
      (contains shot446 cocktail54)
)))
