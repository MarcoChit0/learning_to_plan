(define (problem prob)
 (:domain barman)
 (:objects 
      shaker483 - shaker
      left right - hand
      shot309 - shot
      ingredient325 ingredient89 ingredient429 - ingredient
      cocktail1 - cocktail
      dispenser142 dispenser103 dispenser204 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker483)
  (ontable shot309)
  (dispenses dispenser142 ingredient325)
  (dispenses dispenser103 ingredient89)
  (dispenses dispenser204 ingredient429)
  (clean shaker483)
  (clean shot309)
  (empty shaker483)
  (empty shot309)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker483 l0)
  (shaker-level shaker483 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient325)
  (cocktail-part2 cocktail1 ingredient429)
)
 (:goal
  (and
      (contains shot309 cocktail1)
)))
