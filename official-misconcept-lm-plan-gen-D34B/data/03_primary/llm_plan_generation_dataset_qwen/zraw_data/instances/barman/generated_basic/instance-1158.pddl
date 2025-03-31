(define (problem prob)
 (:domain barman)
 (:objects 
      shaker354 - shaker
      left right - hand
      shot53 shot345 - shot
      ingredient258 ingredient37 ingredient393 ingredient42 - ingredient
      cocktail1 - cocktail
      dispenser361 dispenser406 dispenser130 dispenser129 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker354)
  (ontable shot53)
  (ontable shot345)
  (dispenses dispenser361 ingredient258)
  (dispenses dispenser406 ingredient37)
  (dispenses dispenser130 ingredient393)
  (dispenses dispenser129 ingredient42)
  (clean shaker354)
  (clean shot53)
  (clean shot345)
  (empty shaker354)
  (empty shot53)
  (empty shot345)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker354 l0)
  (shaker-level shaker354 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient37)
  (cocktail-part2 cocktail1 ingredient393)
)
 (:goal
  (and
      (contains shot53 cocktail1)
)))
