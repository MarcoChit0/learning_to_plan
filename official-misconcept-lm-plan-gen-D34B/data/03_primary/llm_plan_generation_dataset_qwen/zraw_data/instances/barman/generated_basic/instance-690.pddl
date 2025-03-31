(define (problem prob)
 (:domain barman)
 (:objects 
      shaker247 - shaker
      left right - hand
      shot386 shot22 - shot
      ingredient279 ingredient54 ingredient476 - ingredient
      cocktail1 - cocktail
      dispenser260 dispenser385 dispenser175 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker247)
  (ontable shot386)
  (ontable shot22)
  (dispenses dispenser260 ingredient279)
  (dispenses dispenser385 ingredient54)
  (dispenses dispenser175 ingredient476)
  (clean shaker247)
  (clean shot386)
  (clean shot22)
  (empty shaker247)
  (empty shot386)
  (empty shot22)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker247 l0)
  (shaker-level shaker247 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient476)
  (cocktail-part2 cocktail1 ingredient54)
)
 (:goal
  (and
      (contains shot386 cocktail1)
)))
