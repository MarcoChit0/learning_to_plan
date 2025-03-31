(define (problem prob)
 (:domain barman)
 (:objects 
      shaker50 - shaker
      left right - hand
      shot203 shot205 - shot
      ingredient151 ingredient441 ingredient478 ingredient389 - ingredient
      cocktail384 - cocktail
      dispenser486 dispenser288 dispenser87 dispenser253 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker50)
  (ontable shot203)
  (ontable shot205)
  (dispenses dispenser486 ingredient151)
  (dispenses dispenser288 ingredient441)
  (dispenses dispenser87 ingredient478)
  (dispenses dispenser253 ingredient389)
  (clean shaker50)
  (clean shot203)
  (clean shot205)
  (empty shaker50)
  (empty shot203)
  (empty shot205)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker50 l0)
  (shaker-level shaker50 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail384 ingredient389)
  (cocktail-part2 cocktail384 ingredient151)
)
 (:goal
  (and
      (contains shot203 cocktail384)
)))
