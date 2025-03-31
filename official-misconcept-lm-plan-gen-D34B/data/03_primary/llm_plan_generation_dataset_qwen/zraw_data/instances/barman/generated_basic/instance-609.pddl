(define (problem prob)
 (:domain barman)
 (:objects 
      shaker55 - shaker
      left right - hand
      shot469 - shot
      ingredient50 ingredient485 ingredient243 ingredient75 - ingredient
      cocktail1 - cocktail
      dispenser388 dispenser277 dispenser417 dispenser356 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker55)
  (ontable shot469)
  (dispenses dispenser388 ingredient50)
  (dispenses dispenser277 ingredient485)
  (dispenses dispenser417 ingredient243)
  (dispenses dispenser356 ingredient75)
  (clean shaker55)
  (clean shot469)
  (empty shaker55)
  (empty shot469)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker55 l0)
  (shaker-level shaker55 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient75)
  (cocktail-part2 cocktail1 ingredient485)
)
 (:goal
  (and
      (contains shot469 cocktail1)
)))
