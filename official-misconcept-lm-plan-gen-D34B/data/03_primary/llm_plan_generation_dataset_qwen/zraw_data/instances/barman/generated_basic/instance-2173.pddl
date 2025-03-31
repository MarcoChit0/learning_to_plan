(define (problem prob)
 (:domain barman)
 (:objects 
      shaker247 - shaker
      left right - hand
      shot108 shot154 - shot
      ingredient369 ingredient192 - ingredient
      cocktail352 - cocktail
      dispenser251 dispenser418 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker247)
  (ontable shot108)
  (ontable shot154)
  (dispenses dispenser251 ingredient369)
  (dispenses dispenser418 ingredient192)
  (clean shaker247)
  (clean shot108)
  (clean shot154)
  (empty shaker247)
  (empty shot108)
  (empty shot154)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker247 l0)
  (shaker-level shaker247 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail352 ingredient192)
  (cocktail-part2 cocktail352 ingredient369)
)
 (:goal
  (and
      (contains shot108 cocktail352)
)))
