(define (problem prob)
 (:domain barman)
 (:objects 
      shaker154 - shaker
      left right - hand
      shot4 shot394 - shot
      ingredient394 ingredient250 - ingredient
      cocktail1 - cocktail
      dispenser216 dispenser429 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker154)
  (ontable shot4)
  (ontable shot394)
  (dispenses dispenser216 ingredient394)
  (dispenses dispenser429 ingredient250)
  (clean shaker154)
  (clean shot4)
  (clean shot394)
  (empty shaker154)
  (empty shot4)
  (empty shot394)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker154 l0)
  (shaker-level shaker154 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient394)
  (cocktail-part2 cocktail1 ingredient250)
)
 (:goal
  (and
      (contains shot4 cocktail1)
)))
