(define (problem prob)
 (:domain barman)
 (:objects 
      shaker338 - shaker
      left right - hand
      shot202 shot465 - shot
      ingredient171 ingredient387 ingredient153 ingredient299 - ingredient
      cocktail70 - cocktail
      dispenser276 dispenser117 dispenser327 dispenser408 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker338)
  (ontable shot202)
  (ontable shot465)
  (dispenses dispenser276 ingredient171)
  (dispenses dispenser117 ingredient387)
  (dispenses dispenser327 ingredient153)
  (dispenses dispenser408 ingredient299)
  (clean shaker338)
  (clean shot202)
  (clean shot465)
  (empty shaker338)
  (empty shot202)
  (empty shot465)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker338 l0)
  (shaker-level shaker338 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail70 ingredient153)
  (cocktail-part2 cocktail70 ingredient299)
)
 (:goal
  (and
      (contains shot202 cocktail70)
)))
