(define (problem prob)
 (:domain barman)
 (:objects 
      shaker202 - shaker
      left right - hand
      shot252 - shot
      ingredient387 ingredient24 ingredient39 - ingredient
      cocktail316 - cocktail
      dispenser160 dispenser183 dispenser75 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker202)
  (ontable shot252)
  (dispenses dispenser160 ingredient387)
  (dispenses dispenser183 ingredient24)
  (dispenses dispenser75 ingredient39)
  (clean shaker202)
  (clean shot252)
  (empty shaker202)
  (empty shot252)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker202 l0)
  (shaker-level shaker202 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail316 ingredient24)
  (cocktail-part2 cocktail316 ingredient39)
)
 (:goal
  (and
      (contains shot252 cocktail316)
)))
