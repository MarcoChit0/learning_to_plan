(define (problem prob)
 (:domain barman)
 (:objects 
      shaker239 - shaker
      left right - hand
      shot212 shot242 - shot
      ingredient310 ingredient24 - ingredient
      cocktail1 - cocktail
      dispenser432 dispenser476 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker239)
  (ontable shot212)
  (ontable shot242)
  (dispenses dispenser432 ingredient310)
  (dispenses dispenser476 ingredient24)
  (clean shaker239)
  (clean shot212)
  (clean shot242)
  (empty shaker239)
  (empty shot212)
  (empty shot242)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker239 l0)
  (shaker-level shaker239 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient24)
  (cocktail-part2 cocktail1 ingredient310)
)
 (:goal
  (and
      (contains shot212 cocktail1)
)))
