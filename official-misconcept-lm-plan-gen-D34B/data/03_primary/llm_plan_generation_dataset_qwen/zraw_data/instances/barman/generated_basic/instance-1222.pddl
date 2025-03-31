(define (problem prob)
 (:domain barman)
 (:objects 
      shaker228 - shaker
      left right - hand
      shot247 shot130 shot129 - shot
      ingredient148 ingredient219 ingredient45 - ingredient
      cocktail250 - cocktail
      dispenser165 dispenser66 dispenser124 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker228)
  (ontable shot247)
  (ontable shot130)
  (ontable shot129)
  (dispenses dispenser165 ingredient148)
  (dispenses dispenser66 ingredient219)
  (dispenses dispenser124 ingredient45)
  (clean shaker228)
  (clean shot247)
  (clean shot130)
  (clean shot129)
  (empty shaker228)
  (empty shot247)
  (empty shot130)
  (empty shot129)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker228 l0)
  (shaker-level shaker228 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail250 ingredient148)
  (cocktail-part2 cocktail250 ingredient219)
)
 (:goal
  (and
      (contains shot247 cocktail250)
      (contains shot130 ingredient45)
)))
