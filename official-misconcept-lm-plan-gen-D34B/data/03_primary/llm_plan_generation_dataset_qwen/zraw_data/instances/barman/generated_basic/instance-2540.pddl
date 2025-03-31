(define (problem prob)
 (:domain barman)
 (:objects 
      shaker296 - shaker
      left right - hand
      shot111 shot477 - shot
      ingredient183 ingredient37 - ingredient
      cocktail276 - cocktail
      dispenser84 dispenser156 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker296)
  (ontable shot111)
  (ontable shot477)
  (dispenses dispenser84 ingredient183)
  (dispenses dispenser156 ingredient37)
  (clean shaker296)
  (clean shot111)
  (clean shot477)
  (empty shaker296)
  (empty shot111)
  (empty shot477)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker296 l0)
  (shaker-level shaker296 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail276 ingredient183)
  (cocktail-part2 cocktail276 ingredient37)
)
 (:goal
  (and
      (contains shot111 cocktail276)
)))
