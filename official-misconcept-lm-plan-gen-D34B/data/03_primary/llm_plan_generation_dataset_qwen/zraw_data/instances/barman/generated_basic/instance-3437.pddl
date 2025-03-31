(define (problem prob)
 (:domain barman)
 (:objects 
      shaker1 - shaker
      left right - hand
      shot162 shot470 - shot
      ingredient174 ingredient13 ingredient467 ingredient499 - ingredient
      cocktail83 - cocktail
      dispenser442 dispenser178 dispenser147 dispenser360 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker1)
  (ontable shot162)
  (ontable shot470)
  (dispenses dispenser442 ingredient174)
  (dispenses dispenser178 ingredient13)
  (dispenses dispenser147 ingredient467)
  (dispenses dispenser360 ingredient499)
  (clean shaker1)
  (clean shot162)
  (clean shot470)
  (empty shaker1)
  (empty shot162)
  (empty shot470)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker1 l0)
  (shaker-level shaker1 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail83 ingredient467)
  (cocktail-part2 cocktail83 ingredient13)
)
 (:goal
  (and
      (contains shot162 cocktail83)
)))
