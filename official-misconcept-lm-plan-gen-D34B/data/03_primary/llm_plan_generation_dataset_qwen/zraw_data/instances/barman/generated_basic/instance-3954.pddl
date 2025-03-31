(define (problem prob)
 (:domain barman)
 (:objects 
      shaker204 - shaker
      left right - hand
      shot317 shot228 - shot
      ingredient289 ingredient66 ingredient202 ingredient328 - ingredient
      cocktail63 - cocktail
      dispenser335 dispenser333 dispenser125 dispenser200 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker204)
  (ontable shot317)
  (ontable shot228)
  (dispenses dispenser335 ingredient289)
  (dispenses dispenser333 ingredient66)
  (dispenses dispenser125 ingredient202)
  (dispenses dispenser200 ingredient328)
  (clean shaker204)
  (clean shot317)
  (clean shot228)
  (empty shaker204)
  (empty shot317)
  (empty shot228)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker204 l0)
  (shaker-level shaker204 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail63 ingredient289)
  (cocktail-part2 cocktail63 ingredient328)
)
 (:goal
  (and
      (contains shot317 cocktail63)
)))
