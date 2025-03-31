(define (problem prob)
 (:domain barman)
 (:objects 
      shaker426 - shaker
      left right - hand
      shot129 shot15 shot460 - shot
      ingredient61 ingredient247 ingredient344 ingredient203 - ingredient
      cocktail15 - cocktail
      dispenser127 dispenser62 dispenser174 dispenser414 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker426)
  (ontable shot129)
  (ontable shot15)
  (ontable shot460)
  (dispenses dispenser127 ingredient61)
  (dispenses dispenser62 ingredient247)
  (dispenses dispenser174 ingredient344)
  (dispenses dispenser414 ingredient203)
  (clean shaker426)
  (clean shot129)
  (clean shot15)
  (clean shot460)
  (empty shaker426)
  (empty shot129)
  (empty shot15)
  (empty shot460)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker426 l0)
  (shaker-level shaker426 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail15 ingredient203)
  (cocktail-part2 cocktail15 ingredient344)
)
 (:goal
  (and
      (contains shot129 cocktail15)
      (contains shot15 ingredient203)
)))
