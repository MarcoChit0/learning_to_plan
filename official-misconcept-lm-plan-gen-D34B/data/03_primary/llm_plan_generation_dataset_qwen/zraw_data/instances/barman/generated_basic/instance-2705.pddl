(define (problem prob)
 (:domain barman)
 (:objects 
      shaker393 - shaker
      left right - hand
      shot5 shot335 shot322 - shot
      ingredient91 ingredient320 - ingredient
      cocktail270 - cocktail
      dispenser359 dispenser292 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker393)
  (ontable shot5)
  (ontable shot335)
  (ontable shot322)
  (dispenses dispenser359 ingredient91)
  (dispenses dispenser292 ingredient320)
  (clean shaker393)
  (clean shot5)
  (clean shot335)
  (clean shot322)
  (empty shaker393)
  (empty shot5)
  (empty shot335)
  (empty shot322)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker393 l0)
  (shaker-level shaker393 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail270 ingredient320)
  (cocktail-part2 cocktail270 ingredient91)
)
 (:goal
  (and
      (contains shot5 cocktail270)
      (contains shot335 cocktail270)
)))
