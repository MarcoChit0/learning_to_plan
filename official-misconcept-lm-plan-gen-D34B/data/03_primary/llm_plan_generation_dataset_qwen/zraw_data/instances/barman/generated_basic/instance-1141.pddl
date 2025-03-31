(define (problem prob)
 (:domain barman)
 (:objects 
      shaker316 - shaker
      left right - hand
      shot74 shot327 - shot
      ingredient196 ingredient236 ingredient27 - ingredient
      cocktail1 - cocktail
      dispenser226 dispenser469 dispenser476 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker316)
  (ontable shot74)
  (ontable shot327)
  (dispenses dispenser226 ingredient196)
  (dispenses dispenser469 ingredient236)
  (dispenses dispenser476 ingredient27)
  (clean shaker316)
  (clean shot74)
  (clean shot327)
  (empty shaker316)
  (empty shot74)
  (empty shot327)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker316 l0)
  (shaker-level shaker316 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient27)
  (cocktail-part2 cocktail1 ingredient196)
)
 (:goal
  (and
      (contains shot74 cocktail1)
)))
