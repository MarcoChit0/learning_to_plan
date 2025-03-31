(define (problem prob)
 (:domain barman)
 (:objects 
      shaker351 - shaker
      left right - hand
      shot391 shot15 - shot
      ingredient331 ingredient176 ingredient366 ingredient34 - ingredient
      cocktail1 - cocktail
      dispenser10 dispenser115 dispenser493 dispenser201 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker351)
  (ontable shot391)
  (ontable shot15)
  (dispenses dispenser10 ingredient331)
  (dispenses dispenser115 ingredient176)
  (dispenses dispenser493 ingredient366)
  (dispenses dispenser201 ingredient34)
  (clean shaker351)
  (clean shot391)
  (clean shot15)
  (empty shaker351)
  (empty shot391)
  (empty shot15)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker351 l0)
  (shaker-level shaker351 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient366)
  (cocktail-part2 cocktail1 ingredient331)
)
 (:goal
  (and
      (contains shot391 cocktail1)
)))
