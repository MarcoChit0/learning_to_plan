(define (problem prob)
 (:domain barman)
 (:objects 
      shaker332 - shaker
      left right - hand
      shot455 - shot
      ingredient62 ingredient52 ingredient368 - ingredient
      cocktail1 - cocktail
      dispenser366 dispenser493 dispenser68 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker332)
  (ontable shot455)
  (dispenses dispenser366 ingredient62)
  (dispenses dispenser493 ingredient52)
  (dispenses dispenser68 ingredient368)
  (clean shaker332)
  (clean shot455)
  (empty shaker332)
  (empty shot455)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker332 l0)
  (shaker-level shaker332 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient62)
  (cocktail-part2 cocktail1 ingredient52)
)
 (:goal
  (and
      (contains shot455 cocktail1)
)))
