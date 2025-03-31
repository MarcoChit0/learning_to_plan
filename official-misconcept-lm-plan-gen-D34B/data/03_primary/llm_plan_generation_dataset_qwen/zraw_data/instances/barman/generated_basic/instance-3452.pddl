(define (problem prob)
 (:domain barman)
 (:objects 
      shaker33 - shaker
      left right - hand
      shot163 shot13 - shot
      ingredient371 ingredient76 ingredient140 ingredient125 - ingredient
      cocktail385 - cocktail
      dispenser30 dispenser75 dispenser321 dispenser319 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker33)
  (ontable shot163)
  (ontable shot13)
  (dispenses dispenser30 ingredient371)
  (dispenses dispenser75 ingredient76)
  (dispenses dispenser321 ingredient140)
  (dispenses dispenser319 ingredient125)
  (clean shaker33)
  (clean shot163)
  (clean shot13)
  (empty shaker33)
  (empty shot163)
  (empty shot13)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker33 l0)
  (shaker-level shaker33 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail385 ingredient140)
  (cocktail-part2 cocktail385 ingredient371)
)
 (:goal
  (and
      (contains shot163 cocktail385)
)))
