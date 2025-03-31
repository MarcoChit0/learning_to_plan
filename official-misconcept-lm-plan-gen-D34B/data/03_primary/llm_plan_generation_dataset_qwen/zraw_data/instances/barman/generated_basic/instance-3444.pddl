(define (problem prob)
 (:domain barman)
 (:objects 
      shaker150 - shaker
      left right - hand
      shot441 shot404 - shot
      ingredient337 ingredient166 ingredient359 ingredient21 - ingredient
      cocktail22 - cocktail
      dispenser491 dispenser409 dispenser83 dispenser384 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker150)
  (ontable shot441)
  (ontable shot404)
  (dispenses dispenser491 ingredient337)
  (dispenses dispenser409 ingredient166)
  (dispenses dispenser83 ingredient359)
  (dispenses dispenser384 ingredient21)
  (clean shaker150)
  (clean shot441)
  (clean shot404)
  (empty shaker150)
  (empty shot441)
  (empty shot404)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker150 l0)
  (shaker-level shaker150 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail22 ingredient359)
  (cocktail-part2 cocktail22 ingredient21)
)
 (:goal
  (and
      (contains shot441 cocktail22)
)))
