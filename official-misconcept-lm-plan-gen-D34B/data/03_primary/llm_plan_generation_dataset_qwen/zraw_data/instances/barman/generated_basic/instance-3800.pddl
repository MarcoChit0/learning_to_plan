(define (problem prob)
 (:domain barman)
 (:objects 
      shaker273 - shaker
      left right - hand
      shot202 - shot
      ingredient333 ingredient261 ingredient455 ingredient178 - ingredient
      cocktail300 - cocktail
      dispenser463 dispenser51 dispenser416 dispenser474 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker273)
  (ontable shot202)
  (dispenses dispenser463 ingredient333)
  (dispenses dispenser51 ingredient261)
  (dispenses dispenser416 ingredient455)
  (dispenses dispenser474 ingredient178)
  (clean shaker273)
  (clean shot202)
  (empty shaker273)
  (empty shot202)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker273 l0)
  (shaker-level shaker273 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail300 ingredient261)
  (cocktail-part2 cocktail300 ingredient178)
)
 (:goal
  (and
      (contains shot202 cocktail300)
)))
