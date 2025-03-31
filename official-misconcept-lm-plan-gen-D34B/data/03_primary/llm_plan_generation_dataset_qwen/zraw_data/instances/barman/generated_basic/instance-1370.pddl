(define (problem prob)
 (:domain barman)
 (:objects 
      shaker338 - shaker
      left right - hand
      shot19 - shot
      ingredient193 ingredient420 ingredient119 - ingredient
      cocktail74 - cocktail
      dispenser261 dispenser370 dispenser116 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker338)
  (ontable shot19)
  (dispenses dispenser261 ingredient193)
  (dispenses dispenser370 ingredient420)
  (dispenses dispenser116 ingredient119)
  (clean shaker338)
  (clean shot19)
  (empty shaker338)
  (empty shot19)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker338 l0)
  (shaker-level shaker338 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail74 ingredient119)
  (cocktail-part2 cocktail74 ingredient420)
)
 (:goal
  (and
      (contains shot19 cocktail74)
)))
