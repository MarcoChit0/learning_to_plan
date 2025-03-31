(define (problem prob)
 (:domain barman)
 (:objects 
      shaker71 - shaker
      left right - hand
      shot248 shot486 - shot
      ingredient341 ingredient202 ingredient369 ingredient92 - ingredient
      cocktail352 - cocktail
      dispenser41 dispenser331 dispenser183 dispenser161 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker71)
  (ontable shot248)
  (ontable shot486)
  (dispenses dispenser41 ingredient341)
  (dispenses dispenser331 ingredient202)
  (dispenses dispenser183 ingredient369)
  (dispenses dispenser161 ingredient92)
  (clean shaker71)
  (clean shot248)
  (clean shot486)
  (empty shaker71)
  (empty shot248)
  (empty shot486)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker71 l0)
  (shaker-level shaker71 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail352 ingredient202)
  (cocktail-part2 cocktail352 ingredient369)
)
 (:goal
  (and
      (contains shot248 cocktail352)
)))
