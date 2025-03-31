(define (problem prob)
 (:domain barman)
 (:objects 
      shaker497 - shaker
      left right - hand
      shot354 shot136 - shot
      ingredient479 ingredient483 ingredient398 - ingredient
      cocktail73 - cocktail
      dispenser212 dispenser304 dispenser477 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker497)
  (ontable shot354)
  (ontable shot136)
  (dispenses dispenser212 ingredient479)
  (dispenses dispenser304 ingredient483)
  (dispenses dispenser477 ingredient398)
  (clean shaker497)
  (clean shot354)
  (clean shot136)
  (empty shaker497)
  (empty shot354)
  (empty shot136)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker497 l0)
  (shaker-level shaker497 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail73 ingredient398)
  (cocktail-part2 cocktail73 ingredient483)
)
 (:goal
  (and
      (contains shot354 cocktail73)
)))
