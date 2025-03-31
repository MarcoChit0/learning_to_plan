(define (problem prob)
 (:domain barman)
 (:objects 
      shaker312 - shaker
      left right - hand
      shot310 shot166 - shot
      ingredient247 ingredient144 - ingredient
      cocktail1 - cocktail
      dispenser477 dispenser28 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker312)
  (ontable shot310)
  (ontable shot166)
  (dispenses dispenser477 ingredient247)
  (dispenses dispenser28 ingredient144)
  (clean shaker312)
  (clean shot310)
  (clean shot166)
  (empty shaker312)
  (empty shot310)
  (empty shot166)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker312 l0)
  (shaker-level shaker312 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient144)
  (cocktail-part2 cocktail1 ingredient247)
)
 (:goal
  (and
      (contains shot310 cocktail1)
)))
