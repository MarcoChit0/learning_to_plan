(define (problem prob)
 (:domain barman)
 (:objects 
      shaker477 - shaker
      left right - hand
      shot244 shot423 - shot
      ingredient70 ingredient328 ingredient225 ingredient155 - ingredient
      cocktail1 - cocktail
      dispenser5 dispenser219 dispenser182 dispenser439 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker477)
  (ontable shot244)
  (ontable shot423)
  (dispenses dispenser5 ingredient70)
  (dispenses dispenser219 ingredient328)
  (dispenses dispenser182 ingredient225)
  (dispenses dispenser439 ingredient155)
  (clean shaker477)
  (clean shot244)
  (clean shot423)
  (empty shaker477)
  (empty shot244)
  (empty shot423)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker477 l0)
  (shaker-level shaker477 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient70)
  (cocktail-part2 cocktail1 ingredient225)
)
 (:goal
  (and
      (contains shot244 cocktail1)
)))
