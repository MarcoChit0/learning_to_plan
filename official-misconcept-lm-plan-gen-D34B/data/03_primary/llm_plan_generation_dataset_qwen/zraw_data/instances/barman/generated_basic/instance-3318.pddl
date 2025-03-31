(define (problem prob)
 (:domain barman)
 (:objects 
      shaker438 - shaker
      left right - hand
      shot284 shot266 shot461 - shot
      ingredient360 ingredient6 ingredient11 - ingredient
      cocktail285 - cocktail
      dispenser395 dispenser90 dispenser79 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker438)
  (ontable shot284)
  (ontable shot266)
  (ontable shot461)
  (dispenses dispenser395 ingredient360)
  (dispenses dispenser90 ingredient6)
  (dispenses dispenser79 ingredient11)
  (clean shaker438)
  (clean shot284)
  (clean shot266)
  (clean shot461)
  (empty shaker438)
  (empty shot284)
  (empty shot266)
  (empty shot461)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker438 l0)
  (shaker-level shaker438 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail285 ingredient11)
  (cocktail-part2 cocktail285 ingredient6)
)
 (:goal
  (and
      (contains shot284 cocktail285)
      (contains shot266 cocktail285)
)))
