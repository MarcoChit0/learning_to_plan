(define (problem prob)
 (:domain barman)
 (:objects 
      shaker34 - shaker
      left right - hand
      shot6 shot164 - shot
      ingredient306 ingredient390 - ingredient
      cocktail1 - cocktail
      dispenser436 dispenser136 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker34)
  (ontable shot6)
  (ontable shot164)
  (dispenses dispenser436 ingredient306)
  (dispenses dispenser136 ingredient390)
  (clean shaker34)
  (clean shot6)
  (clean shot164)
  (empty shaker34)
  (empty shot6)
  (empty shot164)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker34 l0)
  (shaker-level shaker34 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient306)
  (cocktail-part2 cocktail1 ingredient390)
)
 (:goal
  (and
      (contains shot6 cocktail1)
)))
