(define (problem prob)
 (:domain barman)
 (:objects 
      shaker233 - shaker
      left right - hand
      shot477 shot90 shot168 shot397 - shot
      ingredient335 ingredient100 - ingredient
      cocktail172 - cocktail
      dispenser319 dispenser47 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker233)
  (ontable shot477)
  (ontable shot90)
  (ontable shot168)
  (ontable shot397)
  (dispenses dispenser319 ingredient335)
  (dispenses dispenser47 ingredient100)
  (clean shaker233)
  (clean shot477)
  (clean shot90)
  (clean shot168)
  (clean shot397)
  (empty shaker233)
  (empty shot477)
  (empty shot90)
  (empty shot168)
  (empty shot397)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker233 l0)
  (shaker-level shaker233 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail172 ingredient335)
  (cocktail-part2 cocktail172 ingredient100)
)
 (:goal
  (and
      (contains shot477 cocktail172)
      (contains shot90 cocktail172)
      (contains shot168 ingredient335)
)))
