(define (problem prob)
 (:domain barman)
 (:objects 
      shaker236 - shaker
      left right - hand
      shot465 - shot
      ingredient113 ingredient453 ingredient107 ingredient471 - ingredient
      cocktail446 - cocktail
      dispenser490 dispenser422 dispenser259 dispenser40 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker236)
  (ontable shot465)
  (dispenses dispenser490 ingredient113)
  (dispenses dispenser422 ingredient453)
  (dispenses dispenser259 ingredient107)
  (dispenses dispenser40 ingredient471)
  (clean shaker236)
  (clean shot465)
  (empty shaker236)
  (empty shot465)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker236 l0)
  (shaker-level shaker236 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail446 ingredient113)
  (cocktail-part2 cocktail446 ingredient471)
)
 (:goal
  (and
      (contains shot465 cocktail446)
)))
