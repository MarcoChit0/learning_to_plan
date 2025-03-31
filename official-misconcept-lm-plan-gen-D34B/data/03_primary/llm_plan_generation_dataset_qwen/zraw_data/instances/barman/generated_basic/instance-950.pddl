(define (problem prob)
 (:domain barman)
 (:objects 
      shaker172 - shaker
      left right - hand
      shot19 - shot
      ingredient291 ingredient239 ingredient384 ingredient22 - ingredient
      cocktail1 - cocktail
      dispenser47 dispenser294 dispenser248 dispenser364 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker172)
  (ontable shot19)
  (dispenses dispenser47 ingredient291)
  (dispenses dispenser294 ingredient239)
  (dispenses dispenser248 ingredient384)
  (dispenses dispenser364 ingredient22)
  (clean shaker172)
  (clean shot19)
  (empty shaker172)
  (empty shot19)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker172 l0)
  (shaker-level shaker172 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient22)
  (cocktail-part2 cocktail1 ingredient384)
)
 (:goal
  (and
      (contains shot19 cocktail1)
)))
