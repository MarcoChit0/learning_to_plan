(define (problem prob)
 (:domain barman)
 (:objects 
      shaker288 - shaker
      left right - hand
      shot115 shot366 - shot
      ingredient441 ingredient214 ingredient461 - ingredient
      cocktail179 - cocktail
      dispenser458 dispenser2 dispenser269 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker288)
  (ontable shot115)
  (ontable shot366)
  (dispenses dispenser458 ingredient441)
  (dispenses dispenser2 ingredient214)
  (dispenses dispenser269 ingredient461)
  (clean shaker288)
  (clean shot115)
  (clean shot366)
  (empty shaker288)
  (empty shot115)
  (empty shot366)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker288 l0)
  (shaker-level shaker288 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail179 ingredient461)
  (cocktail-part2 cocktail179 ingredient214)
)
 (:goal
  (and
      (contains shot115 cocktail179)
)))
