(define (problem prob)
 (:domain barman)
 (:objects 
      shaker322 - shaker
      left right - hand
      shot253 - shot
      ingredient368 ingredient73 - ingredient
      cocktail111 - cocktail
      dispenser16 dispenser284 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker322)
  (ontable shot253)
  (dispenses dispenser16 ingredient368)
  (dispenses dispenser284 ingredient73)
  (clean shaker322)
  (clean shot253)
  (empty shaker322)
  (empty shot253)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker322 l0)
  (shaker-level shaker322 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail111 ingredient368)
  (cocktail-part2 cocktail111 ingredient73)
)
 (:goal
  (and
      (contains shot253 cocktail111)
)))
