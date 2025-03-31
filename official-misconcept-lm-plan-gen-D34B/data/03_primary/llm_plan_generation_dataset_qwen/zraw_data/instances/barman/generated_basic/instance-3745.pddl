(define (problem prob)
 (:domain barman)
 (:objects 
      shaker340 - shaker
      left right - hand
      shot493 shot404 shot304 - shot
      ingredient365 ingredient288 ingredient107 - ingredient
      cocktail299 - cocktail
      dispenser77 dispenser384 dispenser179 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker340)
  (ontable shot493)
  (ontable shot404)
  (ontable shot304)
  (dispenses dispenser77 ingredient365)
  (dispenses dispenser384 ingredient288)
  (dispenses dispenser179 ingredient107)
  (clean shaker340)
  (clean shot493)
  (clean shot404)
  (clean shot304)
  (empty shaker340)
  (empty shot493)
  (empty shot404)
  (empty shot304)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker340 l0)
  (shaker-level shaker340 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail299 ingredient365)
  (cocktail-part2 cocktail299 ingredient107)
)
 (:goal
  (and
      (contains shot493 cocktail299)
      (contains shot404 ingredient288)
)))
