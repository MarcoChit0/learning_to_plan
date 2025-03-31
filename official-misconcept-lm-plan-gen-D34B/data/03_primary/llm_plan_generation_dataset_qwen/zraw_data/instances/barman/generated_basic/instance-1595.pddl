(define (problem prob)
 (:domain barman)
 (:objects 
      shaker108 - shaker
      left right - hand
      shot138 shot238 - shot
      ingredient449 ingredient134 - ingredient
      cocktail116 - cocktail
      dispenser364 dispenser296 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker108)
  (ontable shot138)
  (ontable shot238)
  (dispenses dispenser364 ingredient449)
  (dispenses dispenser296 ingredient134)
  (clean shaker108)
  (clean shot138)
  (clean shot238)
  (empty shaker108)
  (empty shot138)
  (empty shot238)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker108 l0)
  (shaker-level shaker108 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail116 ingredient449)
  (cocktail-part2 cocktail116 ingredient134)
)
 (:goal
  (and
      (contains shot138 cocktail116)
)))
