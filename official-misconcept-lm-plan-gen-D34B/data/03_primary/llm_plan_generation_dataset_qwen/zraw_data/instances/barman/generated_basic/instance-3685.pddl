(define (problem prob)
 (:domain barman)
 (:objects 
      shaker48 - shaker
      left right - hand
      shot306 shot19 shot162 - shot
      ingredient112 ingredient92 ingredient376 ingredient288 - ingredient
      cocktail282 - cocktail
      dispenser160 dispenser118 dispenser401 dispenser189 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker48)
  (ontable shot306)
  (ontable shot19)
  (ontable shot162)
  (dispenses dispenser160 ingredient112)
  (dispenses dispenser118 ingredient92)
  (dispenses dispenser401 ingredient376)
  (dispenses dispenser189 ingredient288)
  (clean shaker48)
  (clean shot306)
  (clean shot19)
  (clean shot162)
  (empty shaker48)
  (empty shot306)
  (empty shot19)
  (empty shot162)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker48 l0)
  (shaker-level shaker48 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail282 ingredient376)
  (cocktail-part2 cocktail282 ingredient112)
)
 (:goal
  (and
      (contains shot306 cocktail282)
      (contains shot19 ingredient112)
)))
