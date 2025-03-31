(define (problem prob)
 (:domain barman)
 (:objects 
      shaker352 - shaker
      left right - hand
      shot342 shot239 - shot
      ingredient111 ingredient14 ingredient466 ingredient39 - ingredient
      cocktail301 - cocktail
      dispenser461 dispenser279 dispenser88 dispenser319 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker352)
  (ontable shot342)
  (ontable shot239)
  (dispenses dispenser461 ingredient111)
  (dispenses dispenser279 ingredient14)
  (dispenses dispenser88 ingredient466)
  (dispenses dispenser319 ingredient39)
  (clean shaker352)
  (clean shot342)
  (clean shot239)
  (empty shaker352)
  (empty shot342)
  (empty shot239)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker352 l0)
  (shaker-level shaker352 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail301 ingredient14)
  (cocktail-part2 cocktail301 ingredient39)
)
 (:goal
  (and
      (contains shot342 cocktail301)
)))
