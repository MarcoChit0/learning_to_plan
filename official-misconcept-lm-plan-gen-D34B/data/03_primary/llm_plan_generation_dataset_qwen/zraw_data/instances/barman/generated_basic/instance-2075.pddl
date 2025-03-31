(define (problem prob)
 (:domain barman)
 (:objects 
      shaker498 - shaker
      left right - hand
      shot356 shot0 - shot
      ingredient461 ingredient72 - ingredient
      cocktail457 - cocktail
      dispenser349 dispenser328 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker498)
  (ontable shot356)
  (ontable shot0)
  (dispenses dispenser349 ingredient461)
  (dispenses dispenser328 ingredient72)
  (clean shaker498)
  (clean shot356)
  (clean shot0)
  (empty shaker498)
  (empty shot356)
  (empty shot0)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker498 l0)
  (shaker-level shaker498 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail457 ingredient461)
  (cocktail-part2 cocktail457 ingredient72)
)
 (:goal
  (and
      (contains shot356 cocktail457)
)))
