(define (problem prob)
 (:domain barman)
 (:objects 
      shaker183 - shaker
      left right - hand
      shot90 shot288 shot210 - shot
      ingredient397 ingredient55 - ingredient
      cocktail484 - cocktail
      dispenser94 dispenser132 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker183)
  (ontable shot90)
  (ontable shot288)
  (ontable shot210)
  (dispenses dispenser94 ingredient397)
  (dispenses dispenser132 ingredient55)
  (clean shaker183)
  (clean shot90)
  (clean shot288)
  (clean shot210)
  (empty shaker183)
  (empty shot90)
  (empty shot288)
  (empty shot210)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker183 l0)
  (shaker-level shaker183 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail484 ingredient397)
  (cocktail-part2 cocktail484 ingredient55)
)
 (:goal
  (and
      (contains shot90 cocktail484)
      (contains shot288 cocktail484)
)))
