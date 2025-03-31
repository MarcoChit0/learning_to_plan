(define (problem prob)
 (:domain barman)
 (:objects 
      shaker203 - shaker
      left right - hand
      shot193 - shot
      ingredient364 ingredient229 ingredient11 ingredient75 - ingredient
      cocktail273 - cocktail
      dispenser367 dispenser300 dispenser307 dispenser217 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker203)
  (ontable shot193)
  (dispenses dispenser367 ingredient364)
  (dispenses dispenser300 ingredient229)
  (dispenses dispenser307 ingredient11)
  (dispenses dispenser217 ingredient75)
  (clean shaker203)
  (clean shot193)
  (empty shaker203)
  (empty shot193)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker203 l0)
  (shaker-level shaker203 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail273 ingredient364)
  (cocktail-part2 cocktail273 ingredient11)
)
 (:goal
  (and
      (contains shot193 cocktail273)
)))
