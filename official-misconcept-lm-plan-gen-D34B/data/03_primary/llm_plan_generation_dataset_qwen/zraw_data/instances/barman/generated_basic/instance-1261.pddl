(define (problem prob)
 (:domain barman)
 (:objects 
      shaker150 - shaker
      left right - hand
      shot236 shot97 shot21 - shot
      ingredient330 ingredient355 ingredient80 ingredient292 - ingredient
      cocktail236 - cocktail
      dispenser215 dispenser90 dispenser198 dispenser131 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker150)
  (ontable shot236)
  (ontable shot97)
  (ontable shot21)
  (dispenses dispenser215 ingredient330)
  (dispenses dispenser90 ingredient355)
  (dispenses dispenser198 ingredient80)
  (dispenses dispenser131 ingredient292)
  (clean shaker150)
  (clean shot236)
  (clean shot97)
  (clean shot21)
  (empty shaker150)
  (empty shot236)
  (empty shot97)
  (empty shot21)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker150 l0)
  (shaker-level shaker150 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail236 ingredient80)
  (cocktail-part2 cocktail236 ingredient355)
)
 (:goal
  (and
      (contains shot236 cocktail236)
      (contains shot97 ingredient330)
)))
