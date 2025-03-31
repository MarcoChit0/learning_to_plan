(define (problem prob)
 (:domain barman)
 (:objects 
      shaker18 - shaker
      left right - hand
      shot423 shot486 - shot
      ingredient459 ingredient398 - ingredient
      cocktail397 - cocktail
      dispenser211 dispenser35 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker18)
  (ontable shot423)
  (ontable shot486)
  (dispenses dispenser211 ingredient459)
  (dispenses dispenser35 ingredient398)
  (clean shaker18)
  (clean shot423)
  (clean shot486)
  (empty shaker18)
  (empty shot423)
  (empty shot486)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker18 l0)
  (shaker-level shaker18 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail397 ingredient459)
  (cocktail-part2 cocktail397 ingredient398)
)
 (:goal
  (and
      (contains shot423 cocktail397)
)))
