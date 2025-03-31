(define (problem prob)
 (:domain barman)
 (:objects 
      shaker195 - shaker
      left right - hand
      shot454 - shot
      ingredient84 ingredient430 ingredient401 - ingredient
      cocktail251 - cocktail
      dispenser6 dispenser346 dispenser300 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker195)
  (ontable shot454)
  (dispenses dispenser6 ingredient84)
  (dispenses dispenser346 ingredient430)
  (dispenses dispenser300 ingredient401)
  (clean shaker195)
  (clean shot454)
  (empty shaker195)
  (empty shot454)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker195 l0)
  (shaker-level shaker195 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail251 ingredient430)
  (cocktail-part2 cocktail251 ingredient84)
)
 (:goal
  (and
      (contains shot454 cocktail251)
)))
