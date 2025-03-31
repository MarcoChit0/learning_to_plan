(define (problem prob)
 (:domain barman)
 (:objects 
      shaker441 - shaker
      left right - hand
      shot192 shot256 - shot
      ingredient32 ingredient13 - ingredient
      cocktail1 - cocktail
      dispenser21 dispenser134 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker441)
  (ontable shot192)
  (ontable shot256)
  (dispenses dispenser21 ingredient32)
  (dispenses dispenser134 ingredient13)
  (clean shaker441)
  (clean shot192)
  (clean shot256)
  (empty shaker441)
  (empty shot192)
  (empty shot256)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker441 l0)
  (shaker-level shaker441 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient32)
  (cocktail-part2 cocktail1 ingredient13)
)
 (:goal
  (and
      (contains shot192 cocktail1)
)))
