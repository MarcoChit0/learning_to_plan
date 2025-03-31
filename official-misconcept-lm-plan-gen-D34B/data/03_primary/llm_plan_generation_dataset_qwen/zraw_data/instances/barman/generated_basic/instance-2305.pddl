(define (problem prob)
 (:domain barman)
 (:objects 
      shaker338 - shaker
      left right - hand
      shot102 - shot
      ingredient482 ingredient287 - ingredient
      cocktail18 - cocktail
      dispenser180 dispenser330 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker338)
  (ontable shot102)
  (dispenses dispenser180 ingredient482)
  (dispenses dispenser330 ingredient287)
  (clean shaker338)
  (clean shot102)
  (empty shaker338)
  (empty shot102)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker338 l0)
  (shaker-level shaker338 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail18 ingredient482)
  (cocktail-part2 cocktail18 ingredient287)
)
 (:goal
  (and
      (contains shot102 cocktail18)
)))
