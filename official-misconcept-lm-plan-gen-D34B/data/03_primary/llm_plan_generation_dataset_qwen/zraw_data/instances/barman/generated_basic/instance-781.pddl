(define (problem prob)
 (:domain barman)
 (:objects 
      shaker119 - shaker
      left right - hand
      shot49 - shot
      ingredient320 ingredient353 - ingredient
      cocktail1 - cocktail
      dispenser39 dispenser202 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker119)
  (ontable shot49)
  (dispenses dispenser39 ingredient320)
  (dispenses dispenser202 ingredient353)
  (clean shaker119)
  (clean shot49)
  (empty shaker119)
  (empty shot49)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker119 l0)
  (shaker-level shaker119 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient320)
  (cocktail-part2 cocktail1 ingredient353)
)
 (:goal
  (and
      (contains shot49 cocktail1)
)))
