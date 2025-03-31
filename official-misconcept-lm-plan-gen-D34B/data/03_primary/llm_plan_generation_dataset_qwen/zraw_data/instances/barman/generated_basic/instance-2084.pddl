(define (problem prob)
 (:domain barman)
 (:objects 
      shaker162 - shaker
      left right - hand
      shot430 shot113 - shot
      ingredient414 ingredient41 - ingredient
      cocktail256 - cocktail
      dispenser232 dispenser204 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker162)
  (ontable shot430)
  (ontable shot113)
  (dispenses dispenser232 ingredient414)
  (dispenses dispenser204 ingredient41)
  (clean shaker162)
  (clean shot430)
  (clean shot113)
  (empty shaker162)
  (empty shot430)
  (empty shot113)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker162 l0)
  (shaker-level shaker162 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail256 ingredient41)
  (cocktail-part2 cocktail256 ingredient414)
)
 (:goal
  (and
      (contains shot430 cocktail256)
)))
