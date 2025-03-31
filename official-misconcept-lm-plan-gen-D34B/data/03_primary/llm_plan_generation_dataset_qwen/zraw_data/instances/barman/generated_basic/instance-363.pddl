(define (problem prob)
 (:domain barman)
 (:objects 
      shaker72 - shaker
      left right - hand
      shot82 - shot
      ingredient323 ingredient353 - ingredient
      cocktail1 - cocktail
      dispenser369 dispenser483 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker72)
  (ontable shot82)
  (dispenses dispenser369 ingredient323)
  (dispenses dispenser483 ingredient353)
  (clean shaker72)
  (clean shot82)
  (empty shaker72)
  (empty shot82)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker72 l0)
  (shaker-level shaker72 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient353)
  (cocktail-part2 cocktail1 ingredient323)
)
 (:goal
  (and
      (contains shot82 cocktail1)
)))
