(define (problem prob)
 (:domain barman)
 (:objects 
      shaker238 - shaker
      left right - hand
      shot215 shot334 shot318 - shot
      ingredient259 ingredient387 ingredient67 ingredient365 - ingredient
      cocktail426 - cocktail
      dispenser429 dispenser447 dispenser462 dispenser475 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker238)
  (ontable shot215)
  (ontable shot334)
  (ontable shot318)
  (dispenses dispenser429 ingredient259)
  (dispenses dispenser447 ingredient387)
  (dispenses dispenser462 ingredient67)
  (dispenses dispenser475 ingredient365)
  (clean shaker238)
  (clean shot215)
  (clean shot334)
  (clean shot318)
  (empty shaker238)
  (empty shot215)
  (empty shot334)
  (empty shot318)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker238 l0)
  (shaker-level shaker238 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail426 ingredient259)
  (cocktail-part2 cocktail426 ingredient67)
)
 (:goal
  (and
      (contains shot215 cocktail426)
      (contains shot334 cocktail426)
)))
