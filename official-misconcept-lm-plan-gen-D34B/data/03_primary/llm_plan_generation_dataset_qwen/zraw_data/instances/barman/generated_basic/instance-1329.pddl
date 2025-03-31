(define (problem prob)
 (:domain barman)
 (:objects 
      shaker482 - shaker
      left right - hand
      shot116 - shot
      ingredient463 ingredient321 ingredient206 ingredient428 - ingredient
      cocktail458 - cocktail
      dispenser381 dispenser292 dispenser168 dispenser174 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker482)
  (ontable shot116)
  (dispenses dispenser381 ingredient463)
  (dispenses dispenser292 ingredient321)
  (dispenses dispenser168 ingredient206)
  (dispenses dispenser174 ingredient428)
  (clean shaker482)
  (clean shot116)
  (empty shaker482)
  (empty shot116)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker482 l0)
  (shaker-level shaker482 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail458 ingredient428)
  (cocktail-part2 cocktail458 ingredient321)
)
 (:goal
  (and
      (contains shot116 cocktail458)
)))
