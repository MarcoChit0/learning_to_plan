(define (problem prob)
 (:domain barman)
 (:objects 
      shaker189 - shaker
      left right - hand
      shot181 shot328 - shot
      ingredient125 ingredient130 ingredient89 - ingredient
      cocktail1 - cocktail
      dispenser39 dispenser494 dispenser109 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker189)
  (ontable shot181)
  (ontable shot328)
  (dispenses dispenser39 ingredient125)
  (dispenses dispenser494 ingredient130)
  (dispenses dispenser109 ingredient89)
  (clean shaker189)
  (clean shot181)
  (clean shot328)
  (empty shaker189)
  (empty shot181)
  (empty shot328)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker189 l0)
  (shaker-level shaker189 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient130)
  (cocktail-part2 cocktail1 ingredient125)
)
 (:goal
  (and
      (contains shot181 cocktail1)
)))
