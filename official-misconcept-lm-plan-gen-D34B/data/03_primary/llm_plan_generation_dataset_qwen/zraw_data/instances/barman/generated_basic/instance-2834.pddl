(define (problem prob)
 (:domain barman)
 (:objects 
      shaker428 - shaker
      left right - hand
      shot10 shot387 shot134 - shot
      ingredient300 ingredient267 ingredient344 ingredient130 - ingredient
      cocktail221 - cocktail
      dispenser90 dispenser427 dispenser454 dispenser194 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker428)
  (ontable shot10)
  (ontable shot387)
  (ontable shot134)
  (dispenses dispenser90 ingredient300)
  (dispenses dispenser427 ingredient267)
  (dispenses dispenser454 ingredient344)
  (dispenses dispenser194 ingredient130)
  (clean shaker428)
  (clean shot10)
  (clean shot387)
  (clean shot134)
  (empty shaker428)
  (empty shot10)
  (empty shot387)
  (empty shot134)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker428 l0)
  (shaker-level shaker428 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail221 ingredient300)
  (cocktail-part2 cocktail221 ingredient130)
)
 (:goal
  (and
      (contains shot10 cocktail221)
      (contains shot387 ingredient130)
)))
