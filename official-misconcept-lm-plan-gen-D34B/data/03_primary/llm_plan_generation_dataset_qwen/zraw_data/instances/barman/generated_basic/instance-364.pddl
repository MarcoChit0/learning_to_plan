(define (problem prob)
 (:domain barman)
 (:objects 
      shaker67 - shaker
      left right - hand
      shot93 - shot
      ingredient58 ingredient488 - ingredient
      cocktail1 - cocktail
      dispenser364 dispenser461 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker67)
  (ontable shot93)
  (dispenses dispenser364 ingredient58)
  (dispenses dispenser461 ingredient488)
  (clean shaker67)
  (clean shot93)
  (empty shaker67)
  (empty shot93)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker67 l0)
  (shaker-level shaker67 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient58)
  (cocktail-part2 cocktail1 ingredient488)
)
 (:goal
  (and
      (contains shot93 cocktail1)
)))
