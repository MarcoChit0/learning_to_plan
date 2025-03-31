(define (problem prob)
 (:domain barman)
 (:objects 
      shaker61 - shaker
      left right - hand
      shot31 - shot
      ingredient318 ingredient351 - ingredient
      cocktail129 - cocktail
      dispenser169 dispenser236 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker61)
  (ontable shot31)
  (dispenses dispenser169 ingredient318)
  (dispenses dispenser236 ingredient351)
  (clean shaker61)
  (clean shot31)
  (empty shaker61)
  (empty shot31)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker61 l0)
  (shaker-level shaker61 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail129 ingredient318)
  (cocktail-part2 cocktail129 ingredient351)
)
 (:goal
  (and
      (contains shot31 cocktail129)
)))
