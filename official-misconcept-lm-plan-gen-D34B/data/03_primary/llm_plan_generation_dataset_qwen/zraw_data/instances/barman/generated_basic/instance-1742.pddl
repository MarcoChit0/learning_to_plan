(define (problem prob)
 (:domain barman)
 (:objects 
      shaker186 - shaker
      left right - hand
      shot365 shot149 - shot
      ingredient13 ingredient193 ingredient284 - ingredient
      cocktail35 - cocktail
      dispenser72 dispenser212 dispenser461 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker186)
  (ontable shot365)
  (ontable shot149)
  (dispenses dispenser72 ingredient13)
  (dispenses dispenser212 ingredient193)
  (dispenses dispenser461 ingredient284)
  (clean shaker186)
  (clean shot365)
  (clean shot149)
  (empty shaker186)
  (empty shot365)
  (empty shot149)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker186 l0)
  (shaker-level shaker186 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail35 ingredient13)
  (cocktail-part2 cocktail35 ingredient284)
)
 (:goal
  (and
      (contains shot365 cocktail35)
)))
