(define (problem prob)
 (:domain barman)
 (:objects 
      shaker483 - shaker
      left right - hand
      shot287 - shot
      ingredient172 ingredient258 - ingredient
      cocktail272 - cocktail
      dispenser397 dispenser267 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker483)
  (ontable shot287)
  (dispenses dispenser397 ingredient172)
  (dispenses dispenser267 ingredient258)
  (clean shaker483)
  (clean shot287)
  (empty shaker483)
  (empty shot287)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker483 l0)
  (shaker-level shaker483 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail272 ingredient258)
  (cocktail-part2 cocktail272 ingredient172)
)
 (:goal
  (and
      (contains shot287 cocktail272)
)))
