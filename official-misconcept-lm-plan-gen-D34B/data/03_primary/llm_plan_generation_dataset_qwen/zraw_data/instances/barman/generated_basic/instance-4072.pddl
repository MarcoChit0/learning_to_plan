(define (problem prob)
 (:domain barman)
 (:objects 
      shaker432 - shaker
      left right - hand
      shot285 shot323 - shot
      ingredient391 ingredient388 - ingredient
      cocktail343 - cocktail
      dispenser106 dispenser240 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker432)
  (ontable shot285)
  (ontable shot323)
  (dispenses dispenser106 ingredient391)
  (dispenses dispenser240 ingredient388)
  (clean shaker432)
  (clean shot285)
  (clean shot323)
  (empty shaker432)
  (empty shot285)
  (empty shot323)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker432 l0)
  (shaker-level shaker432 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail343 ingredient391)
  (cocktail-part2 cocktail343 ingredient388)
)
 (:goal
  (and
      (contains shot285 cocktail343)
)))
