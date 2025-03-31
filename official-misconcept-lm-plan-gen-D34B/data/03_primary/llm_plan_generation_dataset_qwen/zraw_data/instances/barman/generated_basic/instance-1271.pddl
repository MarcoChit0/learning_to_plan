(define (problem prob)
 (:domain barman)
 (:objects 
      shaker270 - shaker
      left right - hand
      shot286 - shot
      ingredient328 ingredient197 ingredient258 ingredient273 - ingredient
      cocktail174 - cocktail
      dispenser394 dispenser242 dispenser172 dispenser30 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker270)
  (ontable shot286)
  (dispenses dispenser394 ingredient328)
  (dispenses dispenser242 ingredient197)
  (dispenses dispenser172 ingredient258)
  (dispenses dispenser30 ingredient273)
  (clean shaker270)
  (clean shot286)
  (empty shaker270)
  (empty shot286)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker270 l0)
  (shaker-level shaker270 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail174 ingredient258)
  (cocktail-part2 cocktail174 ingredient273)
)
 (:goal
  (and
      (contains shot286 cocktail174)
)))
