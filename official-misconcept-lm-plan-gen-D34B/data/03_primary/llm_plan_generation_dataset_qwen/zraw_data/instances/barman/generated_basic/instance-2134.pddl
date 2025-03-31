(define (problem prob)
 (:domain barman)
 (:objects 
      shaker487 - shaker
      left right - hand
      shot178 shot288 shot362 - shot
      ingredient172 ingredient201 - ingredient
      cocktail312 - cocktail
      dispenser347 dispenser272 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker487)
  (ontable shot178)
  (ontable shot288)
  (ontable shot362)
  (dispenses dispenser347 ingredient172)
  (dispenses dispenser272 ingredient201)
  (clean shaker487)
  (clean shot178)
  (clean shot288)
  (clean shot362)
  (empty shaker487)
  (empty shot178)
  (empty shot288)
  (empty shot362)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker487 l0)
  (shaker-level shaker487 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail312 ingredient172)
  (cocktail-part2 cocktail312 ingredient201)
)
 (:goal
  (and
      (contains shot178 cocktail312)
      (contains shot288 ingredient172)
)))
