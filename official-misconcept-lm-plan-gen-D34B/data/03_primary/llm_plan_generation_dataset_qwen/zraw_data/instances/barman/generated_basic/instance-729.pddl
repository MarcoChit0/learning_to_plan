(define (problem prob)
 (:domain barman)
 (:objects 
      shaker387 - shaker
      left right - hand
      shot269 - shot
      ingredient194 ingredient76 ingredient269 - ingredient
      cocktail1 - cocktail
      dispenser381 dispenser304 dispenser356 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker387)
  (ontable shot269)
  (dispenses dispenser381 ingredient194)
  (dispenses dispenser304 ingredient76)
  (dispenses dispenser356 ingredient269)
  (clean shaker387)
  (clean shot269)
  (empty shaker387)
  (empty shot269)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker387 l0)
  (shaker-level shaker387 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient76)
  (cocktail-part2 cocktail1 ingredient194)
)
 (:goal
  (and
      (contains shot269 cocktail1)
)))
