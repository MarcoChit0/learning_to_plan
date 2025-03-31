(define (problem prob)
 (:domain barman)
 (:objects 
      shaker287 - shaker
      left right - hand
      shot256 shot242 - shot
      ingredient252 ingredient71 - ingredient
      cocktail225 - cocktail
      dispenser68 dispenser50 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker287)
  (ontable shot256)
  (ontable shot242)
  (dispenses dispenser68 ingredient252)
  (dispenses dispenser50 ingredient71)
  (clean shaker287)
  (clean shot256)
  (clean shot242)
  (empty shaker287)
  (empty shot256)
  (empty shot242)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker287 l0)
  (shaker-level shaker287 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail225 ingredient252)
  (cocktail-part2 cocktail225 ingredient71)
)
 (:goal
  (and
      (contains shot256 cocktail225)
)))
