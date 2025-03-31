(define (problem prob)
 (:domain barman)
 (:objects 
      shaker336 - shaker
      left right - hand
      shot148 shot491 shot297 - shot
      ingredient473 ingredient388 - ingredient
      cocktail135 - cocktail
      dispenser489 dispenser427 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker336)
  (ontable shot148)
  (ontable shot491)
  (ontable shot297)
  (dispenses dispenser489 ingredient473)
  (dispenses dispenser427 ingredient388)
  (clean shaker336)
  (clean shot148)
  (clean shot491)
  (clean shot297)
  (empty shaker336)
  (empty shot148)
  (empty shot491)
  (empty shot297)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker336 l0)
  (shaker-level shaker336 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail135 ingredient473)
  (cocktail-part2 cocktail135 ingredient388)
)
 (:goal
  (and
      (contains shot148 cocktail135)
      (contains shot491 cocktail135)
)))
