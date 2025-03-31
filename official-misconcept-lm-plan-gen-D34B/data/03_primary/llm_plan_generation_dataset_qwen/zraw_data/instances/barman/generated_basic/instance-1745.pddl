(define (problem prob)
 (:domain barman)
 (:objects 
      shaker247 - shaker
      left right - hand
      shot272 shot182 - shot
      ingredient312 ingredient332 ingredient477 - ingredient
      cocktail490 - cocktail
      dispenser286 dispenser216 dispenser138 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker247)
  (ontable shot272)
  (ontable shot182)
  (dispenses dispenser286 ingredient312)
  (dispenses dispenser216 ingredient332)
  (dispenses dispenser138 ingredient477)
  (clean shaker247)
  (clean shot272)
  (clean shot182)
  (empty shaker247)
  (empty shot272)
  (empty shot182)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker247 l0)
  (shaker-level shaker247 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail490 ingredient312)
  (cocktail-part2 cocktail490 ingredient477)
)
 (:goal
  (and
      (contains shot272 cocktail490)
)))
