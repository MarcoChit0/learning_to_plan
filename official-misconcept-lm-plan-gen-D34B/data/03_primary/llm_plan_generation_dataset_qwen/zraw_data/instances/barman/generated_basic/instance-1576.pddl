(define (problem prob)
 (:domain barman)
 (:objects 
      shaker221 - shaker
      left right - hand
      shot73 shot467 shot91 - shot
      ingredient332 ingredient124 ingredient147 ingredient74 - ingredient
      cocktail370 - cocktail
      dispenser368 dispenser164 dispenser146 dispenser3 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker221)
  (ontable shot73)
  (ontable shot467)
  (ontable shot91)
  (dispenses dispenser368 ingredient332)
  (dispenses dispenser164 ingredient124)
  (dispenses dispenser146 ingredient147)
  (dispenses dispenser3 ingredient74)
  (clean shaker221)
  (clean shot73)
  (clean shot467)
  (clean shot91)
  (empty shaker221)
  (empty shot73)
  (empty shot467)
  (empty shot91)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker221 l0)
  (shaker-level shaker221 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail370 ingredient147)
  (cocktail-part2 cocktail370 ingredient332)
)
 (:goal
  (and
      (contains shot73 cocktail370)
      (contains shot467 cocktail370)
)))
