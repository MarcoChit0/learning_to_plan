(define (problem prob)
 (:domain barman)
 (:objects 
      shaker274 - shaker
      left right - hand
      shot146 shot344 shot351 - shot
      ingredient246 ingredient164 ingredient360 - ingredient
      cocktail36 - cocktail
      dispenser39 dispenser60 dispenser384 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker274)
  (ontable shot146)
  (ontable shot344)
  (ontable shot351)
  (dispenses dispenser39 ingredient246)
  (dispenses dispenser60 ingredient164)
  (dispenses dispenser384 ingredient360)
  (clean shaker274)
  (clean shot146)
  (clean shot344)
  (clean shot351)
  (empty shaker274)
  (empty shot146)
  (empty shot344)
  (empty shot351)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker274 l0)
  (shaker-level shaker274 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail36 ingredient164)
  (cocktail-part2 cocktail36 ingredient246)
)
 (:goal
  (and
      (contains shot146 cocktail36)
      (contains shot344 ingredient360)
)))
