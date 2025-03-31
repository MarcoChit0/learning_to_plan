(define (problem prob)
 (:domain barman)
 (:objects 
      shaker306 - shaker
      left right - hand
      shot257 shot131 shot313 - shot
      ingredient239 ingredient311 - ingredient
      cocktail225 - cocktail
      dispenser18 dispenser209 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker306)
  (ontable shot257)
  (ontable shot131)
  (ontable shot313)
  (dispenses dispenser18 ingredient239)
  (dispenses dispenser209 ingredient311)
  (clean shaker306)
  (clean shot257)
  (clean shot131)
  (clean shot313)
  (empty shaker306)
  (empty shot257)
  (empty shot131)
  (empty shot313)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker306 l0)
  (shaker-level shaker306 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail225 ingredient239)
  (cocktail-part2 cocktail225 ingredient311)
)
 (:goal
  (and
      (contains shot257 cocktail225)
      (contains shot131 ingredient239)
)))
