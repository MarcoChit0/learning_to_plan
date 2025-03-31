(define (problem prob)
 (:domain barman)
 (:objects 
      shaker314 - shaker
      left right - hand
      shot320 - shot
      ingredient70 ingredient44 - ingredient
      cocktail467 - cocktail
      dispenser119 dispenser103 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker314)
  (ontable shot320)
  (dispenses dispenser119 ingredient70)
  (dispenses dispenser103 ingredient44)
  (clean shaker314)
  (clean shot320)
  (empty shaker314)
  (empty shot320)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker314 l0)
  (shaker-level shaker314 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail467 ingredient70)
  (cocktail-part2 cocktail467 ingredient44)
)
 (:goal
  (and
      (contains shot320 cocktail467)
)))
