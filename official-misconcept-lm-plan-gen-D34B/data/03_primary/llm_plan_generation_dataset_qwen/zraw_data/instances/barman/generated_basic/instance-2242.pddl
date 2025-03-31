(define (problem prob)
 (:domain barman)
 (:objects 
      shaker362 - shaker
      left right - hand
      shot444 shot393 shot26 - shot
      ingredient436 ingredient328 ingredient444 ingredient431 - ingredient
      cocktail309 - cocktail
      dispenser370 dispenser284 dispenser149 dispenser120 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker362)
  (ontable shot444)
  (ontable shot393)
  (ontable shot26)
  (dispenses dispenser370 ingredient436)
  (dispenses dispenser284 ingredient328)
  (dispenses dispenser149 ingredient444)
  (dispenses dispenser120 ingredient431)
  (clean shaker362)
  (clean shot444)
  (clean shot393)
  (clean shot26)
  (empty shaker362)
  (empty shot444)
  (empty shot393)
  (empty shot26)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker362 l0)
  (shaker-level shaker362 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail309 ingredient431)
  (cocktail-part2 cocktail309 ingredient328)
)
 (:goal
  (and
      (contains shot444 cocktail309)
      (contains shot393 cocktail309)
)))
