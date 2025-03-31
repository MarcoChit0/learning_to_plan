(define (problem prob)
 (:domain barman)
 (:objects 
      shaker275 - shaker
      left right - hand
      shot111 shot134 shot477 - shot
      ingredient72 ingredient1 - ingredient
      cocktail466 - cocktail
      dispenser457 dispenser474 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker275)
  (ontable shot111)
  (ontable shot134)
  (ontable shot477)
  (dispenses dispenser457 ingredient72)
  (dispenses dispenser474 ingredient1)
  (clean shaker275)
  (clean shot111)
  (clean shot134)
  (clean shot477)
  (empty shaker275)
  (empty shot111)
  (empty shot134)
  (empty shot477)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker275 l0)
  (shaker-level shaker275 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail466 ingredient72)
  (cocktail-part2 cocktail466 ingredient1)
)
 (:goal
  (and
      (contains shot111 cocktail466)
      (contains shot134 cocktail466)
)))
