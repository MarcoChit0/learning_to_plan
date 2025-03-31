(define (problem prob)
 (:domain barman)
 (:objects 
      shaker401 - shaker
      left right - hand
      shot139 shot495 shot461 - shot
      ingredient13 ingredient127 ingredient171 - ingredient
      cocktail397 - cocktail
      dispenser141 dispenser53 dispenser186 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker401)
  (ontable shot139)
  (ontable shot495)
  (ontable shot461)
  (dispenses dispenser141 ingredient13)
  (dispenses dispenser53 ingredient127)
  (dispenses dispenser186 ingredient171)
  (clean shaker401)
  (clean shot139)
  (clean shot495)
  (clean shot461)
  (empty shaker401)
  (empty shot139)
  (empty shot495)
  (empty shot461)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker401 l0)
  (shaker-level shaker401 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail397 ingredient127)
  (cocktail-part2 cocktail397 ingredient13)
)
 (:goal
  (and
      (contains shot139 cocktail397)
      (contains shot495 ingredient171)
)))
