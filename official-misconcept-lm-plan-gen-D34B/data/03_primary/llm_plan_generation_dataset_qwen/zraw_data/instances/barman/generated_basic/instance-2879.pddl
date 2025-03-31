(define (problem prob)
 (:domain barman)
 (:objects 
      shaker368 - shaker
      left right - hand
      shot204 shot3 shot366 - shot
      ingredient427 ingredient343 - ingredient
      cocktail403 - cocktail
      dispenser97 dispenser284 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker368)
  (ontable shot204)
  (ontable shot3)
  (ontable shot366)
  (dispenses dispenser97 ingredient427)
  (dispenses dispenser284 ingredient343)
  (clean shaker368)
  (clean shot204)
  (clean shot3)
  (clean shot366)
  (empty shaker368)
  (empty shot204)
  (empty shot3)
  (empty shot366)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker368 l0)
  (shaker-level shaker368 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail403 ingredient343)
  (cocktail-part2 cocktail403 ingredient427)
)
 (:goal
  (and
      (contains shot204 cocktail403)
      (contains shot3 cocktail403)
)))
