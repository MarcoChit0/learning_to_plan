(define (problem prob)
 (:domain barman)
 (:objects 
      shaker77 - shaker
      left right - hand
      shot228 - shot
      ingredient239 ingredient144 - ingredient
      cocktail34 - cocktail
      dispenser188 dispenser94 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker77)
  (ontable shot228)
  (dispenses dispenser188 ingredient239)
  (dispenses dispenser94 ingredient144)
  (clean shaker77)
  (clean shot228)
  (empty shaker77)
  (empty shot228)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker77 l0)
  (shaker-level shaker77 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail34 ingredient239)
  (cocktail-part2 cocktail34 ingredient144)
)
 (:goal
  (and
      (contains shot228 cocktail34)
)))
