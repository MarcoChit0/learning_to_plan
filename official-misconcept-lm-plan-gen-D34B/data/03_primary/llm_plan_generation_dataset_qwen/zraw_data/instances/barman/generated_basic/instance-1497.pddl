(define (problem prob)
 (:domain barman)
 (:objects 
      shaker221 - shaker
      left right - hand
      shot185 shot96 - shot
      ingredient144 ingredient50 ingredient17 ingredient427 - ingredient
      cocktail237 - cocktail
      dispenser309 dispenser392 dispenser216 dispenser275 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker221)
  (ontable shot185)
  (ontable shot96)
  (dispenses dispenser309 ingredient144)
  (dispenses dispenser392 ingredient50)
  (dispenses dispenser216 ingredient17)
  (dispenses dispenser275 ingredient427)
  (clean shaker221)
  (clean shot185)
  (clean shot96)
  (empty shaker221)
  (empty shot185)
  (empty shot96)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker221 l0)
  (shaker-level shaker221 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail237 ingredient144)
  (cocktail-part2 cocktail237 ingredient17)
)
 (:goal
  (and
      (contains shot185 cocktail237)
)))
