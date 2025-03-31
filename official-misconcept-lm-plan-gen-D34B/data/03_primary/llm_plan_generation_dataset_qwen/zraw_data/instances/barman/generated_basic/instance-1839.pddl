(define (problem prob)
 (:domain barman)
 (:objects 
      shaker314 - shaker
      left right - hand
      shot187 shot109 - shot
      ingredient396 ingredient170 ingredient362 - ingredient
      cocktail28 - cocktail
      dispenser379 dispenser320 dispenser61 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker314)
  (ontable shot187)
  (ontable shot109)
  (dispenses dispenser379 ingredient396)
  (dispenses dispenser320 ingredient170)
  (dispenses dispenser61 ingredient362)
  (clean shaker314)
  (clean shot187)
  (clean shot109)
  (empty shaker314)
  (empty shot187)
  (empty shot109)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker314 l0)
  (shaker-level shaker314 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail28 ingredient170)
  (cocktail-part2 cocktail28 ingredient396)
)
 (:goal
  (and
      (contains shot187 cocktail28)
)))
