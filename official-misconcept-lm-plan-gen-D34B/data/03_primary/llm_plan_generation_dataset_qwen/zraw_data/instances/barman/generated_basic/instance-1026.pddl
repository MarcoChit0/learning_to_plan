(define (problem prob)
 (:domain barman)
 (:objects 
      shaker107 - shaker
      left right - hand
      shot302 - shot
      ingredient129 ingredient176 ingredient295 - ingredient
      cocktail1 - cocktail
      dispenser163 dispenser38 dispenser255 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker107)
  (ontable shot302)
  (dispenses dispenser163 ingredient129)
  (dispenses dispenser38 ingredient176)
  (dispenses dispenser255 ingredient295)
  (clean shaker107)
  (clean shot302)
  (empty shaker107)
  (empty shot302)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker107 l0)
  (shaker-level shaker107 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient176)
  (cocktail-part2 cocktail1 ingredient129)
)
 (:goal
  (and
      (contains shot302 cocktail1)
)))
