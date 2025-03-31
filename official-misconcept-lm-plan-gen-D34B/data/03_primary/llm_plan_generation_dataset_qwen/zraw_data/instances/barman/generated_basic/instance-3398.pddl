(define (problem prob)
 (:domain barman)
 (:objects 
      shaker358 - shaker
      left right - hand
      shot274 shot11 - shot
      ingredient330 ingredient64 ingredient255 ingredient231 - ingredient
      cocktail130 - cocktail
      dispenser199 dispenser41 dispenser98 dispenser474 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker358)
  (ontable shot274)
  (ontable shot11)
  (dispenses dispenser199 ingredient330)
  (dispenses dispenser41 ingredient64)
  (dispenses dispenser98 ingredient255)
  (dispenses dispenser474 ingredient231)
  (clean shaker358)
  (clean shot274)
  (clean shot11)
  (empty shaker358)
  (empty shot274)
  (empty shot11)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker358 l0)
  (shaker-level shaker358 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail130 ingredient255)
  (cocktail-part2 cocktail130 ingredient330)
)
 (:goal
  (and
      (contains shot274 cocktail130)
)))
