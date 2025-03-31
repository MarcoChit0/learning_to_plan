(define (problem prob)
 (:domain barman)
 (:objects 
      shaker121 - shaker
      left right - hand
      shot248 shot459 shot366 - shot
      ingredient216 ingredient435 ingredient48 - ingredient
      cocktail1 - cocktail
      dispenser190 dispenser88 dispenser217 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker121)
  (ontable shot248)
  (ontable shot459)
  (ontable shot366)
  (dispenses dispenser190 ingredient216)
  (dispenses dispenser88 ingredient435)
  (dispenses dispenser217 ingredient48)
  (clean shaker121)
  (clean shot248)
  (clean shot459)
  (clean shot366)
  (empty shaker121)
  (empty shot248)
  (empty shot459)
  (empty shot366)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker121 l0)
  (shaker-level shaker121 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient435)
  (cocktail-part2 cocktail1 ingredient216)
)
 (:goal
  (and
      (contains shot248 cocktail1)
      (contains shot459 cocktail1)
)))
