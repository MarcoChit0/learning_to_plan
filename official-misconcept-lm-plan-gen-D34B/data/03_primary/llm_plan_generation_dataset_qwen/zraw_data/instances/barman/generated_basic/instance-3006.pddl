(define (problem prob)
 (:domain barman)
 (:objects 
      shaker338 - shaker
      left right - hand
      shot430 shot117 - shot
      ingredient404 ingredient136 ingredient457 - ingredient
      cocktail63 - cocktail
      dispenser87 dispenser131 dispenser377 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker338)
  (ontable shot430)
  (ontable shot117)
  (dispenses dispenser87 ingredient404)
  (dispenses dispenser131 ingredient136)
  (dispenses dispenser377 ingredient457)
  (clean shaker338)
  (clean shot430)
  (clean shot117)
  (empty shaker338)
  (empty shot430)
  (empty shot117)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker338 l0)
  (shaker-level shaker338 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail63 ingredient136)
  (cocktail-part2 cocktail63 ingredient404)
)
 (:goal
  (and
      (contains shot430 cocktail63)
)))
