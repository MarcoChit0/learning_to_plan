(define (problem prob)
 (:domain barman)
 (:objects 
      shaker203 - shaker
      left right - hand
      shot295 - shot
      ingredient375 ingredient32 ingredient137 ingredient448 - ingredient
      cocktail21 - cocktail
      dispenser237 dispenser230 dispenser352 dispenser115 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker203)
  (ontable shot295)
  (dispenses dispenser237 ingredient375)
  (dispenses dispenser230 ingredient32)
  (dispenses dispenser352 ingredient137)
  (dispenses dispenser115 ingredient448)
  (clean shaker203)
  (clean shot295)
  (empty shaker203)
  (empty shot295)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker203 l0)
  (shaker-level shaker203 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail21 ingredient137)
  (cocktail-part2 cocktail21 ingredient32)
)
 (:goal
  (and
      (contains shot295 cocktail21)
)))
