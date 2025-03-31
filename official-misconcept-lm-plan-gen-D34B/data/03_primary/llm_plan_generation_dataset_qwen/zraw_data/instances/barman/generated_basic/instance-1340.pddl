(define (problem prob)
 (:domain barman)
 (:objects 
      shaker215 - shaker
      left right - hand
      shot163 shot175 shot391 - shot
      ingredient5 ingredient4 ingredient306 - ingredient
      cocktail388 - cocktail
      dispenser485 dispenser217 dispenser73 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker215)
  (ontable shot163)
  (ontable shot175)
  (ontable shot391)
  (dispenses dispenser485 ingredient5)
  (dispenses dispenser217 ingredient4)
  (dispenses dispenser73 ingredient306)
  (clean shaker215)
  (clean shot163)
  (clean shot175)
  (clean shot391)
  (empty shaker215)
  (empty shot163)
  (empty shot175)
  (empty shot391)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker215 l0)
  (shaker-level shaker215 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail388 ingredient306)
  (cocktail-part2 cocktail388 ingredient5)
)
 (:goal
  (and
      (contains shot163 cocktail388)
      (contains shot175 ingredient306)
)))
