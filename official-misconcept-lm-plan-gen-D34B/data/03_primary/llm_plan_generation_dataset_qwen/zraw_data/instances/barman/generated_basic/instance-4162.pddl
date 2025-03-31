(define (problem prob)
 (:domain barman)
 (:objects 
      shaker467 - shaker
      left right - hand
      shot163 shot162 - shot
      ingredient327 ingredient253 - ingredient
      cocktail0 - cocktail
      dispenser23 dispenser281 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker467)
  (ontable shot163)
  (ontable shot162)
  (dispenses dispenser23 ingredient327)
  (dispenses dispenser281 ingredient253)
  (clean shaker467)
  (clean shot163)
  (clean shot162)
  (empty shaker467)
  (empty shot163)
  (empty shot162)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker467 l0)
  (shaker-level shaker467 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail0 ingredient253)
  (cocktail-part2 cocktail0 ingredient327)
)
 (:goal
  (and
      (contains shot163 cocktail0)
)))
