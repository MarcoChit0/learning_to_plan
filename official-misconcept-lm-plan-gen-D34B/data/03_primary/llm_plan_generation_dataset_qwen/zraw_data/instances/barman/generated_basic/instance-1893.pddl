(define (problem prob)
 (:domain barman)
 (:objects 
      shaker195 - shaker
      left right - hand
      shot339 shot216 - shot
      ingredient360 ingredient330 ingredient293 - ingredient
      cocktail399 - cocktail
      dispenser90 dispenser202 dispenser259 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker195)
  (ontable shot339)
  (ontable shot216)
  (dispenses dispenser90 ingredient360)
  (dispenses dispenser202 ingredient330)
  (dispenses dispenser259 ingredient293)
  (clean shaker195)
  (clean shot339)
  (clean shot216)
  (empty shaker195)
  (empty shot339)
  (empty shot216)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker195 l0)
  (shaker-level shaker195 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail399 ingredient293)
  (cocktail-part2 cocktail399 ingredient360)
)
 (:goal
  (and
      (contains shot339 cocktail399)
)))
