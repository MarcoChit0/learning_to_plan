(define (problem prob)
 (:domain barman)
 (:objects 
      shaker386 - shaker
      left right - hand
      shot387 shot209 - shot
      ingredient195 ingredient345 ingredient332 - ingredient
      cocktail44 - cocktail
      dispenser60 dispenser181 dispenser80 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker386)
  (ontable shot387)
  (ontable shot209)
  (dispenses dispenser60 ingredient195)
  (dispenses dispenser181 ingredient345)
  (dispenses dispenser80 ingredient332)
  (clean shaker386)
  (clean shot387)
  (clean shot209)
  (empty shaker386)
  (empty shot387)
  (empty shot209)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker386 l0)
  (shaker-level shaker386 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail44 ingredient195)
  (cocktail-part2 cocktail44 ingredient345)
)
 (:goal
  (and
      (contains shot387 cocktail44)
)))
