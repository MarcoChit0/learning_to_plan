(define (problem prob)
 (:domain barman)
 (:objects 
      shaker318 - shaker
      left right - hand
      shot135 shot251 - shot
      ingredient79 ingredient433 ingredient292 ingredient457 - ingredient
      cocktail54 - cocktail
      dispenser473 dispenser401 dispenser57 dispenser48 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker318)
  (ontable shot135)
  (ontable shot251)
  (dispenses dispenser473 ingredient79)
  (dispenses dispenser401 ingredient433)
  (dispenses dispenser57 ingredient292)
  (dispenses dispenser48 ingredient457)
  (clean shaker318)
  (clean shot135)
  (clean shot251)
  (empty shaker318)
  (empty shot135)
  (empty shot251)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker318 l0)
  (shaker-level shaker318 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail54 ingredient79)
  (cocktail-part2 cocktail54 ingredient433)
)
 (:goal
  (and
      (contains shot135 cocktail54)
)))
