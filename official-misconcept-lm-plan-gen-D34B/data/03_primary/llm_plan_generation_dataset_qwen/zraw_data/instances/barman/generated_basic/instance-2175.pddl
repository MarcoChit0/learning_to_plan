(define (problem prob)
 (:domain barman)
 (:objects 
      shaker262 - shaker
      left right - hand
      shot344 shot108 - shot
      ingredient408 ingredient465 - ingredient
      cocktail20 - cocktail
      dispenser468 dispenser146 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker262)
  (ontable shot344)
  (ontable shot108)
  (dispenses dispenser468 ingredient408)
  (dispenses dispenser146 ingredient465)
  (clean shaker262)
  (clean shot344)
  (clean shot108)
  (empty shaker262)
  (empty shot344)
  (empty shot108)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker262 l0)
  (shaker-level shaker262 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail20 ingredient408)
  (cocktail-part2 cocktail20 ingredient465)
)
 (:goal
  (and
      (contains shot344 cocktail20)
)))
