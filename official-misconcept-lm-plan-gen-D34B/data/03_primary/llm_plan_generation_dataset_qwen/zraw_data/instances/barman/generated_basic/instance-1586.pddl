(define (problem prob)
 (:domain barman)
 (:objects 
      shaker247 - shaker
      left right - hand
      shot287 shot182 - shot
      ingredient24 ingredient261 - ingredient
      cocktail399 - cocktail
      dispenser253 dispenser396 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker247)
  (ontable shot287)
  (ontable shot182)
  (dispenses dispenser253 ingredient24)
  (dispenses dispenser396 ingredient261)
  (clean shaker247)
  (clean shot287)
  (clean shot182)
  (empty shaker247)
  (empty shot287)
  (empty shot182)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker247 l0)
  (shaker-level shaker247 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail399 ingredient24)
  (cocktail-part2 cocktail399 ingredient261)
)
 (:goal
  (and
      (contains shot287 cocktail399)
)))
