(define (problem prob)
 (:domain barman)
 (:objects 
      shaker303 - shaker
      left right - hand
      shot136 - shot
      ingredient345 ingredient129 ingredient395 ingredient62 - ingredient
      cocktail243 - cocktail
      dispenser101 dispenser201 dispenser466 dispenser222 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker303)
  (ontable shot136)
  (dispenses dispenser101 ingredient345)
  (dispenses dispenser201 ingredient129)
  (dispenses dispenser466 ingredient395)
  (dispenses dispenser222 ingredient62)
  (clean shaker303)
  (clean shot136)
  (empty shaker303)
  (empty shot136)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker303 l0)
  (shaker-level shaker303 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail243 ingredient62)
  (cocktail-part2 cocktail243 ingredient129)
)
 (:goal
  (and
      (contains shot136 cocktail243)
)))
