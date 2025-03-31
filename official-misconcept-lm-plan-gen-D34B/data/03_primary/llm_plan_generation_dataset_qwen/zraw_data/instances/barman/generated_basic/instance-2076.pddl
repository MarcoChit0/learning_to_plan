(define (problem prob)
 (:domain barman)
 (:objects 
      shaker420 - shaker
      left right - hand
      shot446 shot55 - shot
      ingredient35 ingredient144 - ingredient
      cocktail5 - cocktail
      dispenser264 dispenser431 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker420)
  (ontable shot446)
  (ontable shot55)
  (dispenses dispenser264 ingredient35)
  (dispenses dispenser431 ingredient144)
  (clean shaker420)
  (clean shot446)
  (clean shot55)
  (empty shaker420)
  (empty shot446)
  (empty shot55)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker420 l0)
  (shaker-level shaker420 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail5 ingredient144)
  (cocktail-part2 cocktail5 ingredient35)
)
 (:goal
  (and
      (contains shot446 cocktail5)
)))
