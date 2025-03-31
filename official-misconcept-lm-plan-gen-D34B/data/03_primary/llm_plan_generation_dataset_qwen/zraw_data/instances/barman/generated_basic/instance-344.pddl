(define (problem prob)
 (:domain barman)
 (:objects 
      shaker400 - shaker
      left right - hand
      shot167 shot138 - shot
      ingredient346 ingredient311 ingredient415 ingredient420 - ingredient
      cocktail1 - cocktail
      dispenser290 dispenser474 dispenser196 dispenser30 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker400)
  (ontable shot167)
  (ontable shot138)
  (dispenses dispenser290 ingredient346)
  (dispenses dispenser474 ingredient311)
  (dispenses dispenser196 ingredient415)
  (dispenses dispenser30 ingredient420)
  (clean shaker400)
  (clean shot167)
  (clean shot138)
  (empty shaker400)
  (empty shot167)
  (empty shot138)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker400 l0)
  (shaker-level shaker400 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient415)
  (cocktail-part2 cocktail1 ingredient311)
)
 (:goal
  (and
      (contains shot167 cocktail1)
)))
