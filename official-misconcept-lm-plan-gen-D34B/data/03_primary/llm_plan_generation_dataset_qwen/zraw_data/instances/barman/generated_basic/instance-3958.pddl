(define (problem prob)
 (:domain barman)
 (:objects 
      shaker491 - shaker
      left right - hand
      shot147 shot238 - shot
      ingredient315 ingredient427 ingredient225 ingredient196 - ingredient
      cocktail105 - cocktail
      dispenser200 dispenser33 dispenser118 dispenser289 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker491)
  (ontable shot147)
  (ontable shot238)
  (dispenses dispenser200 ingredient315)
  (dispenses dispenser33 ingredient427)
  (dispenses dispenser118 ingredient225)
  (dispenses dispenser289 ingredient196)
  (clean shaker491)
  (clean shot147)
  (clean shot238)
  (empty shaker491)
  (empty shot147)
  (empty shot238)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker491 l0)
  (shaker-level shaker491 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail105 ingredient315)
  (cocktail-part2 cocktail105 ingredient427)
)
 (:goal
  (and
      (contains shot147 cocktail105)
)))
