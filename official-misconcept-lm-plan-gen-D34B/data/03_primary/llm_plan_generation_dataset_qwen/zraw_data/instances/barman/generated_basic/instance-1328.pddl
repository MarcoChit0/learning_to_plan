(define (problem prob)
 (:domain barman)
 (:objects 
      shaker399 - shaker
      left right - hand
      shot191 - shot
      ingredient309 ingredient482 ingredient210 ingredient85 - ingredient
      cocktail368 - cocktail
      dispenser58 dispenser308 dispenser246 dispenser172 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker399)
  (ontable shot191)
  (dispenses dispenser58 ingredient309)
  (dispenses dispenser308 ingredient482)
  (dispenses dispenser246 ingredient210)
  (dispenses dispenser172 ingredient85)
  (clean shaker399)
  (clean shot191)
  (empty shaker399)
  (empty shot191)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker399 l0)
  (shaker-level shaker399 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail368 ingredient482)
  (cocktail-part2 cocktail368 ingredient85)
)
 (:goal
  (and
      (contains shot191 cocktail368)
)))
