(define (problem prob)
 (:domain barman)
 (:objects 
      shaker403 - shaker
      left right - hand
      shot485 - shot
      ingredient487 ingredient491 - ingredient
      cocktail1 - cocktail
      dispenser179 dispenser271 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker403)
  (ontable shot485)
  (dispenses dispenser179 ingredient487)
  (dispenses dispenser271 ingredient491)
  (clean shaker403)
  (clean shot485)
  (empty shaker403)
  (empty shot485)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker403 l0)
  (shaker-level shaker403 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient487)
  (cocktail-part2 cocktail1 ingredient491)
)
 (:goal
  (and
      (contains shot485 cocktail1)
)))
