(define (problem prob)
 (:domain barman)
 (:objects 
      shaker467 - shaker
      left right - hand
      shot26 shot342 - shot
      ingredient352 ingredient347 - ingredient
      cocktail90 - cocktail
      dispenser455 dispenser239 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker467)
  (ontable shot26)
  (ontable shot342)
  (dispenses dispenser455 ingredient352)
  (dispenses dispenser239 ingredient347)
  (clean shaker467)
  (clean shot26)
  (clean shot342)
  (empty shaker467)
  (empty shot26)
  (empty shot342)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker467 l0)
  (shaker-level shaker467 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail90 ingredient352)
  (cocktail-part2 cocktail90 ingredient347)
)
 (:goal
  (and
      (contains shot26 cocktail90)
)))
