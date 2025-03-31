(define (problem prob)
 (:domain barman)
 (:objects 
      shaker302 - shaker
      left right - hand
      shot361 shot73 - shot
      ingredient424 ingredient379 - ingredient
      cocktail334 - cocktail
      dispenser418 dispenser246 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker302)
  (ontable shot361)
  (ontable shot73)
  (dispenses dispenser418 ingredient424)
  (dispenses dispenser246 ingredient379)
  (clean shaker302)
  (clean shot361)
  (clean shot73)
  (empty shaker302)
  (empty shot361)
  (empty shot73)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker302 l0)
  (shaker-level shaker302 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail334 ingredient379)
  (cocktail-part2 cocktail334 ingredient424)
)
 (:goal
  (and
      (contains shot361 cocktail334)
)))
