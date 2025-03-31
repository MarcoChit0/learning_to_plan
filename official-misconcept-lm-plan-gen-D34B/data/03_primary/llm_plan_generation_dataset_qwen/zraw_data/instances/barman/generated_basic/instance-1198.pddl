(define (problem prob)
 (:domain barman)
 (:objects 
      shaker98 - shaker
      left right - hand
      shot140 shot54 - shot
      ingredient410 ingredient137 - ingredient
      cocktail2 - cocktail
      dispenser408 dispenser13 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker98)
  (ontable shot140)
  (ontable shot54)
  (dispenses dispenser408 ingredient410)
  (dispenses dispenser13 ingredient137)
  (clean shaker98)
  (clean shot140)
  (clean shot54)
  (empty shaker98)
  (empty shot140)
  (empty shot54)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker98 l0)
  (shaker-level shaker98 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail2 ingredient410)
  (cocktail-part2 cocktail2 ingredient137)
)
 (:goal
  (and
      (contains shot140 cocktail2)
)))
