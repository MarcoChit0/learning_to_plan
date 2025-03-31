(define (problem prob)
 (:domain barman)
 (:objects 
      shaker326 - shaker
      left right - hand
      shot101 shot27 shot419 - shot
      ingredient8 ingredient356 ingredient43 - ingredient
      cocktail150 - cocktail
      dispenser367 dispenser420 dispenser132 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker326)
  (ontable shot101)
  (ontable shot27)
  (ontable shot419)
  (dispenses dispenser367 ingredient8)
  (dispenses dispenser420 ingredient356)
  (dispenses dispenser132 ingredient43)
  (clean shaker326)
  (clean shot101)
  (clean shot27)
  (clean shot419)
  (empty shaker326)
  (empty shot101)
  (empty shot27)
  (empty shot419)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker326 l0)
  (shaker-level shaker326 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail150 ingredient43)
  (cocktail-part2 cocktail150 ingredient356)
)
 (:goal
  (and
      (contains shot101 cocktail150)
      (contains shot27 cocktail150)
)))
