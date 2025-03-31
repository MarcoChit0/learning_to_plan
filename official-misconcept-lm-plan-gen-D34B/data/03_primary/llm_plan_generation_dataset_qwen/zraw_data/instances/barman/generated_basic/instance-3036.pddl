(define (problem prob)
 (:domain barman)
 (:objects 
      shaker115 - shaker
      left right - hand
      shot154 - shot
      ingredient436 ingredient388 ingredient283 - ingredient
      cocktail378 - cocktail
      dispenser390 dispenser134 dispenser129 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker115)
  (ontable shot154)
  (dispenses dispenser390 ingredient436)
  (dispenses dispenser134 ingredient388)
  (dispenses dispenser129 ingredient283)
  (clean shaker115)
  (clean shot154)
  (empty shaker115)
  (empty shot154)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker115 l0)
  (shaker-level shaker115 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail378 ingredient436)
  (cocktail-part2 cocktail378 ingredient283)
)
 (:goal
  (and
      (contains shot154 cocktail378)
)))
