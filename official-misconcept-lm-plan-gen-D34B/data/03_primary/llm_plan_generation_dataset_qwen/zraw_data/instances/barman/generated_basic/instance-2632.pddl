(define (problem prob)
 (:domain barman)
 (:objects 
      shaker53 - shaker
      left right - hand
      shot6 shot132 - shot
      ingredient41 ingredient224 ingredient320 - ingredient
      cocktail425 - cocktail
      dispenser322 dispenser358 dispenser124 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker53)
  (ontable shot6)
  (ontable shot132)
  (dispenses dispenser322 ingredient41)
  (dispenses dispenser358 ingredient224)
  (dispenses dispenser124 ingredient320)
  (clean shaker53)
  (clean shot6)
  (clean shot132)
  (empty shaker53)
  (empty shot6)
  (empty shot132)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker53 l0)
  (shaker-level shaker53 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail425 ingredient224)
  (cocktail-part2 cocktail425 ingredient41)
)
 (:goal
  (and
      (contains shot6 cocktail425)
)))
