(define (problem prob)
 (:domain barman)
 (:objects 
      shaker253 - shaker
      left right - hand
      shot186 shot335 - shot
      ingredient135 ingredient187 ingredient382 ingredient42 - ingredient
      cocktail476 - cocktail
      dispenser48 dispenser143 dispenser49 dispenser175 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker253)
  (ontable shot186)
  (ontable shot335)
  (dispenses dispenser48 ingredient135)
  (dispenses dispenser143 ingredient187)
  (dispenses dispenser49 ingredient382)
  (dispenses dispenser175 ingredient42)
  (clean shaker253)
  (clean shot186)
  (clean shot335)
  (empty shaker253)
  (empty shot186)
  (empty shot335)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker253 l0)
  (shaker-level shaker253 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail476 ingredient187)
  (cocktail-part2 cocktail476 ingredient42)
)
 (:goal
  (and
      (contains shot186 cocktail476)
)))
