(define (problem prob)
 (:domain barman)
 (:objects 
      shaker163 - shaker
      left right - hand
      shot165 shot88 - shot
      ingredient13 ingredient76 - ingredient
      cocktail203 - cocktail
      dispenser278 dispenser384 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker163)
  (ontable shot165)
  (ontable shot88)
  (dispenses dispenser278 ingredient13)
  (dispenses dispenser384 ingredient76)
  (clean shaker163)
  (clean shot165)
  (clean shot88)
  (empty shaker163)
  (empty shot165)
  (empty shot88)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker163 l0)
  (shaker-level shaker163 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail203 ingredient13)
  (cocktail-part2 cocktail203 ingredient76)
)
 (:goal
  (and
      (contains shot165 cocktail203)
)))
