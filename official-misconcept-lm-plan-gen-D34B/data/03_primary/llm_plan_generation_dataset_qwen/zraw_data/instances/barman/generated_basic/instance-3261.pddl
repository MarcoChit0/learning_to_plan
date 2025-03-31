(define (problem prob)
 (:domain barman)
 (:objects 
      shaker310 - shaker
      left right - hand
      shot54 shot239 shot376 - shot
      ingredient29 ingredient343 ingredient402 ingredient250 - ingredient
      cocktail302 - cocktail
      dispenser312 dispenser136 dispenser415 dispenser410 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker310)
  (ontable shot54)
  (ontable shot239)
  (ontable shot376)
  (dispenses dispenser312 ingredient29)
  (dispenses dispenser136 ingredient343)
  (dispenses dispenser415 ingredient402)
  (dispenses dispenser410 ingredient250)
  (clean shaker310)
  (clean shot54)
  (clean shot239)
  (clean shot376)
  (empty shaker310)
  (empty shot54)
  (empty shot239)
  (empty shot376)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker310 l0)
  (shaker-level shaker310 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail302 ingredient343)
  (cocktail-part2 cocktail302 ingredient402)
)
 (:goal
  (and
      (contains shot54 cocktail302)
      (contains shot239 cocktail302)
)))
