(define (problem prob)
 (:domain barman)
 (:objects 
      shaker297 - shaker
      left right - hand
      shot182 - shot
      ingredient29 ingredient165 ingredient115 ingredient369 - ingredient
      cocktail237 - cocktail
      dispenser317 dispenser206 dispenser410 dispenser23 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker297)
  (ontable shot182)
  (dispenses dispenser317 ingredient29)
  (dispenses dispenser206 ingredient165)
  (dispenses dispenser410 ingredient115)
  (dispenses dispenser23 ingredient369)
  (clean shaker297)
  (clean shot182)
  (empty shaker297)
  (empty shot182)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker297 l0)
  (shaker-level shaker297 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail237 ingredient29)
  (cocktail-part2 cocktail237 ingredient165)
)
 (:goal
  (and
      (contains shot182 cocktail237)
)))
