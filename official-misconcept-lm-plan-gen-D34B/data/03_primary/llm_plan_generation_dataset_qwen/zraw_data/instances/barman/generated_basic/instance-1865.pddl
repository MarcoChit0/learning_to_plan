(define (problem prob)
 (:domain barman)
 (:objects 
      shaker52 - shaker
      left right - hand
      shot216 shot476 - shot
      ingredient357 ingredient34 - ingredient
      cocktail375 - cocktail
      dispenser222 dispenser425 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker52)
  (ontable shot216)
  (ontable shot476)
  (dispenses dispenser222 ingredient357)
  (dispenses dispenser425 ingredient34)
  (clean shaker52)
  (clean shot216)
  (clean shot476)
  (empty shaker52)
  (empty shot216)
  (empty shot476)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker52 l0)
  (shaker-level shaker52 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail375 ingredient34)
  (cocktail-part2 cocktail375 ingredient357)
)
 (:goal
  (and
      (contains shot216 cocktail375)
)))
