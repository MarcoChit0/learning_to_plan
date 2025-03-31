(define (problem prob)
 (:domain barman)
 (:objects 
      shaker405 - shaker
      left right - hand
      shot256 - shot
      ingredient176 ingredient110 ingredient295 - ingredient
      cocktail121 - cocktail
      dispenser266 dispenser248 dispenser226 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker405)
  (ontable shot256)
  (dispenses dispenser266 ingredient176)
  (dispenses dispenser248 ingredient110)
  (dispenses dispenser226 ingredient295)
  (clean shaker405)
  (clean shot256)
  (empty shaker405)
  (empty shot256)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker405 l0)
  (shaker-level shaker405 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail121 ingredient176)
  (cocktail-part2 cocktail121 ingredient295)
)
 (:goal
  (and
      (contains shot256 cocktail121)
)))
