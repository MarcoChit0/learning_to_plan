(define (problem prob)
 (:domain barman)
 (:objects 
      shaker363 - shaker
      left right - hand
      shot166 - shot
      ingredient397 ingredient465 - ingredient
      cocktail251 - cocktail
      dispenser49 dispenser222 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker363)
  (ontable shot166)
  (dispenses dispenser49 ingredient397)
  (dispenses dispenser222 ingredient465)
  (clean shaker363)
  (clean shot166)
  (empty shaker363)
  (empty shot166)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker363 l0)
  (shaker-level shaker363 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail251 ingredient397)
  (cocktail-part2 cocktail251 ingredient465)
)
 (:goal
  (and
      (contains shot166 cocktail251)
)))
