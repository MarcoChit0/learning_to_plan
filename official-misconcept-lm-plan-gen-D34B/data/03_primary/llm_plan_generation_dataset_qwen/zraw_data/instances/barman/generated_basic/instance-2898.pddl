(define (problem prob)
 (:domain barman)
 (:objects 
      shaker206 - shaker
      left right - hand
      shot179 - shot
      ingredient144 ingredient84 - ingredient
      cocktail54 - cocktail
      dispenser327 dispenser342 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker206)
  (ontable shot179)
  (dispenses dispenser327 ingredient144)
  (dispenses dispenser342 ingredient84)
  (clean shaker206)
  (clean shot179)
  (empty shaker206)
  (empty shot179)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker206 l0)
  (shaker-level shaker206 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail54 ingredient144)
  (cocktail-part2 cocktail54 ingredient84)
)
 (:goal
  (and
      (contains shot179 cocktail54)
)))
