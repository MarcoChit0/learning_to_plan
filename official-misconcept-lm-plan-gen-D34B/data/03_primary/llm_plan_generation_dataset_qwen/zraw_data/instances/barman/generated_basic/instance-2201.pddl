(define (problem prob)
 (:domain barman)
 (:objects 
      shaker236 - shaker
      left right - hand
      shot42 shot330 shot351 - shot
      ingredient482 ingredient342 ingredient97 ingredient473 - ingredient
      cocktail356 - cocktail
      dispenser240 dispenser68 dispenser461 dispenser337 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker236)
  (ontable shot42)
  (ontable shot330)
  (ontable shot351)
  (dispenses dispenser240 ingredient482)
  (dispenses dispenser68 ingredient342)
  (dispenses dispenser461 ingredient97)
  (dispenses dispenser337 ingredient473)
  (clean shaker236)
  (clean shot42)
  (clean shot330)
  (clean shot351)
  (empty shaker236)
  (empty shot42)
  (empty shot330)
  (empty shot351)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker236 l0)
  (shaker-level shaker236 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail356 ingredient473)
  (cocktail-part2 cocktail356 ingredient482)
)
 (:goal
  (and
      (contains shot42 cocktail356)
      (contains shot330 cocktail356)
)))
