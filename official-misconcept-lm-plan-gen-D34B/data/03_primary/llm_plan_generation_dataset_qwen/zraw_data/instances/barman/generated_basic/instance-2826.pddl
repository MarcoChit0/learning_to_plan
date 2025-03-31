(define (problem prob)
 (:domain barman)
 (:objects 
      shaker448 - shaker
      left right - hand
      shot433 shot185 shot202 - shot
      ingredient367 ingredient147 ingredient489 ingredient249 - ingredient
      cocktail357 - cocktail
      dispenser35 dispenser41 dispenser63 dispenser395 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker448)
  (ontable shot433)
  (ontable shot185)
  (ontable shot202)
  (dispenses dispenser35 ingredient367)
  (dispenses dispenser41 ingredient147)
  (dispenses dispenser63 ingredient489)
  (dispenses dispenser395 ingredient249)
  (clean shaker448)
  (clean shot433)
  (clean shot185)
  (clean shot202)
  (empty shaker448)
  (empty shot433)
  (empty shot185)
  (empty shot202)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker448 l0)
  (shaker-level shaker448 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail357 ingredient147)
  (cocktail-part2 cocktail357 ingredient489)
)
 (:goal
  (and
      (contains shot433 cocktail357)
      (contains shot185 cocktail357)
)))
