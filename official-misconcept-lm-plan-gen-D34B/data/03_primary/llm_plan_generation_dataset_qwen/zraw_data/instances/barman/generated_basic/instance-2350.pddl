(define (problem prob)
 (:domain barman)
 (:objects 
      shaker39 - shaker
      left right - hand
      shot139 shot238 - shot
      ingredient51 ingredient1 ingredient37 ingredient55 - ingredient
      cocktail261 - cocktail
      dispenser199 dispenser138 dispenser15 dispenser431 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker39)
  (ontable shot139)
  (ontable shot238)
  (dispenses dispenser199 ingredient51)
  (dispenses dispenser138 ingredient1)
  (dispenses dispenser15 ingredient37)
  (dispenses dispenser431 ingredient55)
  (clean shaker39)
  (clean shot139)
  (clean shot238)
  (empty shaker39)
  (empty shot139)
  (empty shot238)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker39 l0)
  (shaker-level shaker39 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail261 ingredient1)
  (cocktail-part2 cocktail261 ingredient55)
)
 (:goal
  (and
      (contains shot139 cocktail261)
)))
