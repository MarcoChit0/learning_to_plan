(define (problem prob)
 (:domain barman)
 (:objects 
      shaker228 - shaker
      left right - hand
      shot267 shot281 - shot
      ingredient248 ingredient127 ingredient114 - ingredient
      cocktail1 - cocktail
      dispenser461 dispenser133 dispenser322 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker228)
  (ontable shot267)
  (ontable shot281)
  (dispenses dispenser461 ingredient248)
  (dispenses dispenser133 ingredient127)
  (dispenses dispenser322 ingredient114)
  (clean shaker228)
  (clean shot267)
  (clean shot281)
  (empty shaker228)
  (empty shot267)
  (empty shot281)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker228 l0)
  (shaker-level shaker228 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient248)
  (cocktail-part2 cocktail1 ingredient114)
)
 (:goal
  (and
      (contains shot267 cocktail1)
)))
