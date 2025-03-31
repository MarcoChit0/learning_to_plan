(define (problem prob)
 (:domain barman)
 (:objects 
      shaker330 - shaker
      left right - hand
      shot362 shot147 shot438 - shot
      ingredient259 ingredient34 - ingredient
      cocktail109 - cocktail
      dispenser24 dispenser207 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker330)
  (ontable shot362)
  (ontable shot147)
  (ontable shot438)
  (dispenses dispenser24 ingredient259)
  (dispenses dispenser207 ingredient34)
  (clean shaker330)
  (clean shot362)
  (clean shot147)
  (clean shot438)
  (empty shaker330)
  (empty shot362)
  (empty shot147)
  (empty shot438)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker330 l0)
  (shaker-level shaker330 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail109 ingredient34)
  (cocktail-part2 cocktail109 ingredient259)
)
 (:goal
  (and
      (contains shot362 cocktail109)
      (contains shot147 cocktail109)
)))
