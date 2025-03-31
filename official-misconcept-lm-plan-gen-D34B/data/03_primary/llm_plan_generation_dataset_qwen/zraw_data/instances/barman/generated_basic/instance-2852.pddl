(define (problem prob)
 (:domain barman)
 (:objects 
      shaker158 - shaker
      left right - hand
      shot373 shot489 - shot
      ingredient124 ingredient125 ingredient215 ingredient302 - ingredient
      cocktail490 - cocktail
      dispenser457 dispenser83 dispenser34 dispenser222 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker158)
  (ontable shot373)
  (ontable shot489)
  (dispenses dispenser457 ingredient124)
  (dispenses dispenser83 ingredient125)
  (dispenses dispenser34 ingredient215)
  (dispenses dispenser222 ingredient302)
  (clean shaker158)
  (clean shot373)
  (clean shot489)
  (empty shaker158)
  (empty shot373)
  (empty shot489)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker158 l0)
  (shaker-level shaker158 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail490 ingredient124)
  (cocktail-part2 cocktail490 ingredient215)
)
 (:goal
  (and
      (contains shot373 cocktail490)
)))
