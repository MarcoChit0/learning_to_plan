(define (problem prob)
 (:domain barman)
 (:objects 
      shaker222 - shaker
      left right - hand
      shot202 shot35 - shot
      ingredient37 ingredient255 ingredient73 ingredient166 - ingredient
      cocktail299 - cocktail
      dispenser301 dispenser347 dispenser167 dispenser423 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker222)
  (ontable shot202)
  (ontable shot35)
  (dispenses dispenser301 ingredient37)
  (dispenses dispenser347 ingredient255)
  (dispenses dispenser167 ingredient73)
  (dispenses dispenser423 ingredient166)
  (clean shaker222)
  (clean shot202)
  (clean shot35)
  (empty shaker222)
  (empty shot202)
  (empty shot35)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker222 l0)
  (shaker-level shaker222 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail299 ingredient37)
  (cocktail-part2 cocktail299 ingredient166)
)
 (:goal
  (and
      (contains shot202 cocktail299)
)))
