(define (problem prob)
 (:domain barman)
 (:objects 
      shaker159 - shaker
      left right - hand
      shot313 shot168 - shot
      ingredient477 ingredient276 ingredient255 - ingredient
      cocktail5 - cocktail
      dispenser43 dispenser384 dispenser322 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker159)
  (ontable shot313)
  (ontable shot168)
  (dispenses dispenser43 ingredient477)
  (dispenses dispenser384 ingredient276)
  (dispenses dispenser322 ingredient255)
  (clean shaker159)
  (clean shot313)
  (clean shot168)
  (empty shaker159)
  (empty shot313)
  (empty shot168)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker159 l0)
  (shaker-level shaker159 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail5 ingredient276)
  (cocktail-part2 cocktail5 ingredient255)
)
 (:goal
  (and
      (contains shot313 cocktail5)
)))
