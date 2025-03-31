(define (problem prob)
 (:domain barman)
 (:objects 
      shaker252 - shaker
      left right - hand
      shot275 shot136 - shot
      ingredient197 ingredient365 - ingredient
      cocktail1 - cocktail
      dispenser387 dispenser296 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker252)
  (ontable shot275)
  (ontable shot136)
  (dispenses dispenser387 ingredient197)
  (dispenses dispenser296 ingredient365)
  (clean shaker252)
  (clean shot275)
  (clean shot136)
  (empty shaker252)
  (empty shot275)
  (empty shot136)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker252 l0)
  (shaker-level shaker252 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient365)
  (cocktail-part2 cocktail1 ingredient197)
)
 (:goal
  (and
      (contains shot275 cocktail1)
)))
