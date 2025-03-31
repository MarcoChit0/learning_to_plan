(define (problem prob)
 (:domain barman)
 (:objects 
      shaker329 - shaker
      left right - hand
      shot298 shot166 shot342 - shot
      ingredient349 ingredient344 - ingredient
      cocktail280 - cocktail
      dispenser218 dispenser186 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker329)
  (ontable shot298)
  (ontable shot166)
  (ontable shot342)
  (dispenses dispenser218 ingredient349)
  (dispenses dispenser186 ingredient344)
  (clean shaker329)
  (clean shot298)
  (clean shot166)
  (clean shot342)
  (empty shaker329)
  (empty shot298)
  (empty shot166)
  (empty shot342)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker329 l0)
  (shaker-level shaker329 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail280 ingredient349)
  (cocktail-part2 cocktail280 ingredient344)
)
 (:goal
  (and
      (contains shot298 cocktail280)
      (contains shot166 ingredient344)
)))
