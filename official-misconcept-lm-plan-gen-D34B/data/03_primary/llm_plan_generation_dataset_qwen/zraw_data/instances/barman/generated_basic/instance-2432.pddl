(define (problem prob)
 (:domain barman)
 (:objects 
      shaker403 - shaker
      left right - hand
      shot10 shot461 shot380 - shot
      ingredient188 ingredient303 ingredient458 - ingredient
      cocktail228 - cocktail
      dispenser480 dispenser437 dispenser124 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker403)
  (ontable shot10)
  (ontable shot461)
  (ontable shot380)
  (dispenses dispenser480 ingredient188)
  (dispenses dispenser437 ingredient303)
  (dispenses dispenser124 ingredient458)
  (clean shaker403)
  (clean shot10)
  (clean shot461)
  (clean shot380)
  (empty shaker403)
  (empty shot10)
  (empty shot461)
  (empty shot380)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker403 l0)
  (shaker-level shaker403 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail228 ingredient303)
  (cocktail-part2 cocktail228 ingredient188)
)
 (:goal
  (and
      (contains shot10 cocktail228)
      (contains shot461 cocktail228)
)))
