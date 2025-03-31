(define (problem prob)
 (:domain barman)
 (:objects 
      shaker286 - shaker
      left right - hand
      shot198 - shot
      ingredient170 ingredient239 ingredient13 ingredient21 - ingredient
      cocktail211 - cocktail
      dispenser402 dispenser166 dispenser87 dispenser433 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker286)
  (ontable shot198)
  (dispenses dispenser402 ingredient170)
  (dispenses dispenser166 ingredient239)
  (dispenses dispenser87 ingredient13)
  (dispenses dispenser433 ingredient21)
  (clean shaker286)
  (clean shot198)
  (empty shaker286)
  (empty shot198)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker286 l0)
  (shaker-level shaker286 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail211 ingredient170)
  (cocktail-part2 cocktail211 ingredient239)
)
 (:goal
  (and
      (contains shot198 cocktail211)
)))
