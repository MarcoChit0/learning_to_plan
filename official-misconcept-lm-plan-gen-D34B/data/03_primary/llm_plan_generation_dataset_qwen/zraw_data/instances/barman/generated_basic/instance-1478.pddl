(define (problem prob)
 (:domain barman)
 (:objects 
      shaker206 - shaker
      left right - hand
      shot328 shot261 - shot
      ingredient14 ingredient462 ingredient390 ingredient55 - ingredient
      cocktail312 - cocktail
      dispenser41 dispenser56 dispenser226 dispenser432 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker206)
  (ontable shot328)
  (ontable shot261)
  (dispenses dispenser41 ingredient14)
  (dispenses dispenser56 ingredient462)
  (dispenses dispenser226 ingredient390)
  (dispenses dispenser432 ingredient55)
  (clean shaker206)
  (clean shot328)
  (clean shot261)
  (empty shaker206)
  (empty shot328)
  (empty shot261)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker206 l0)
  (shaker-level shaker206 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail312 ingredient14)
  (cocktail-part2 cocktail312 ingredient462)
)
 (:goal
  (and
      (contains shot328 cocktail312)
)))
