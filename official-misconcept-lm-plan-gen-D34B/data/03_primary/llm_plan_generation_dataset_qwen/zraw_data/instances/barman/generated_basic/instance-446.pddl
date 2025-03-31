(define (problem prob)
 (:domain barman)
 (:objects 
      shaker55 - shaker
      left right - hand
      shot276 shot48 - shot
      ingredient476 ingredient4 ingredient313 - ingredient
      cocktail1 - cocktail
      dispenser58 dispenser422 dispenser27 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker55)
  (ontable shot276)
  (ontable shot48)
  (dispenses dispenser58 ingredient476)
  (dispenses dispenser422 ingredient4)
  (dispenses dispenser27 ingredient313)
  (clean shaker55)
  (clean shot276)
  (clean shot48)
  (empty shaker55)
  (empty shot276)
  (empty shot48)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker55 l0)
  (shaker-level shaker55 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient4)
  (cocktail-part2 cocktail1 ingredient313)
)
 (:goal
  (and
      (contains shot276 cocktail1)
)))
