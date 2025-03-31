(define (problem prob)
 (:domain barman)
 (:objects 
      shaker174 - shaker
      left right - hand
      shot348 shot488 shot5 - shot
      ingredient343 ingredient256 ingredient454 - ingredient
      cocktail229 - cocktail
      dispenser388 dispenser391 dispenser323 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker174)
  (ontable shot348)
  (ontable shot488)
  (ontable shot5)
  (dispenses dispenser388 ingredient343)
  (dispenses dispenser391 ingredient256)
  (dispenses dispenser323 ingredient454)
  (clean shaker174)
  (clean shot348)
  (clean shot488)
  (clean shot5)
  (empty shaker174)
  (empty shot348)
  (empty shot488)
  (empty shot5)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker174 l0)
  (shaker-level shaker174 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail229 ingredient454)
  (cocktail-part2 cocktail229 ingredient256)
)
 (:goal
  (and
      (contains shot348 cocktail229)
      (contains shot488 ingredient256)
)))
