(define (problem prob)
 (:domain barman)
 (:objects 
      shaker268 - shaker
      left right - hand
      shot430 - shot
      ingredient437 ingredient144 ingredient103 ingredient226 - ingredient
      cocktail100 - cocktail
      dispenser57 dispenser20 dispenser104 dispenser429 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker268)
  (ontable shot430)
  (dispenses dispenser57 ingredient437)
  (dispenses dispenser20 ingredient144)
  (dispenses dispenser104 ingredient103)
  (dispenses dispenser429 ingredient226)
  (clean shaker268)
  (clean shot430)
  (empty shaker268)
  (empty shot430)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker268 l0)
  (shaker-level shaker268 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail100 ingredient103)
  (cocktail-part2 cocktail100 ingredient144)
)
 (:goal
  (and
      (contains shot430 cocktail100)
)))
