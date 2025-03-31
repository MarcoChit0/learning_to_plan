(define (problem prob)
 (:domain barman)
 (:objects 
      shaker208 - shaker
      left right - hand
      shot354 shot467 - shot
      ingredient242 ingredient297 ingredient52 - ingredient
      cocktail1 - cocktail
      dispenser378 dispenser148 dispenser27 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker208)
  (ontable shot354)
  (ontable shot467)
  (dispenses dispenser378 ingredient242)
  (dispenses dispenser148 ingredient297)
  (dispenses dispenser27 ingredient52)
  (clean shaker208)
  (clean shot354)
  (clean shot467)
  (empty shaker208)
  (empty shot354)
  (empty shot467)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker208 l0)
  (shaker-level shaker208 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient242)
  (cocktail-part2 cocktail1 ingredient297)
)
 (:goal
  (and
      (contains shot354 cocktail1)
)))
