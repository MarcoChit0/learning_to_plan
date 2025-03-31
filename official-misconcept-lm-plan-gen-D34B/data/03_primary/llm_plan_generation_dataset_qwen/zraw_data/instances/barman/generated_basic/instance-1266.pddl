(define (problem prob)
 (:domain barman)
 (:objects 
      shaker120 - shaker
      left right - hand
      shot262 shot109 shot252 - shot
      ingredient66 ingredient219 ingredient95 ingredient67 - ingredient
      cocktail131 - cocktail
      dispenser416 dispenser154 dispenser192 dispenser83 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker120)
  (ontable shot262)
  (ontable shot109)
  (ontable shot252)
  (dispenses dispenser416 ingredient66)
  (dispenses dispenser154 ingredient219)
  (dispenses dispenser192 ingredient95)
  (dispenses dispenser83 ingredient67)
  (clean shaker120)
  (clean shot262)
  (clean shot109)
  (clean shot252)
  (empty shaker120)
  (empty shot262)
  (empty shot109)
  (empty shot252)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker120 l0)
  (shaker-level shaker120 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail131 ingredient67)
  (cocktail-part2 cocktail131 ingredient219)
)
 (:goal
  (and
      (contains shot262 cocktail131)
      (contains shot109 cocktail131)
)))
