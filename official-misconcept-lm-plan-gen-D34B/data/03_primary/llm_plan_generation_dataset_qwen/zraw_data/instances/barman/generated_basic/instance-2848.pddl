(define (problem prob)
 (:domain barman)
 (:objects 
      shaker43 - shaker
      left right - hand
      shot357 shot456 - shot
      ingredient330 ingredient365 ingredient25 ingredient17 - ingredient
      cocktail60 - cocktail
      dispenser20 dispenser80 dispenser451 dispenser64 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker43)
  (ontable shot357)
  (ontable shot456)
  (dispenses dispenser20 ingredient330)
  (dispenses dispenser80 ingredient365)
  (dispenses dispenser451 ingredient25)
  (dispenses dispenser64 ingredient17)
  (clean shaker43)
  (clean shot357)
  (clean shot456)
  (empty shaker43)
  (empty shot357)
  (empty shot456)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker43 l0)
  (shaker-level shaker43 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail60 ingredient25)
  (cocktail-part2 cocktail60 ingredient17)
)
 (:goal
  (and
      (contains shot357 cocktail60)
)))
