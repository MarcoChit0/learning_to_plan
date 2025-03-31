(define (problem prob)
 (:domain barman)
 (:objects 
      shaker494 - shaker
      left right - hand
      shot342 shot341 - shot
      ingredient57 ingredient368 - ingredient
      cocktail1 - cocktail
      dispenser293 dispenser368 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker494)
  (ontable shot342)
  (ontable shot341)
  (dispenses dispenser293 ingredient57)
  (dispenses dispenser368 ingredient368)
  (clean shaker494)
  (clean shot342)
  (clean shot341)
  (empty shaker494)
  (empty shot342)
  (empty shot341)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker494 l0)
  (shaker-level shaker494 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient57)
  (cocktail-part2 cocktail1 ingredient368)
)
 (:goal
  (and
      (contains shot342 cocktail1)
)))
