(define (problem prob)
 (:domain barman)
 (:objects 
      shaker464 - shaker
      left right - hand
      shot394 shot142 shot103 - shot
      ingredient127 ingredient283 - ingredient
      cocktail222 - cocktail
      dispenser89 dispenser316 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker464)
  (ontable shot394)
  (ontable shot142)
  (ontable shot103)
  (dispenses dispenser89 ingredient127)
  (dispenses dispenser316 ingredient283)
  (clean shaker464)
  (clean shot394)
  (clean shot142)
  (clean shot103)
  (empty shaker464)
  (empty shot394)
  (empty shot142)
  (empty shot103)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker464 l0)
  (shaker-level shaker464 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail222 ingredient283)
  (cocktail-part2 cocktail222 ingredient127)
)
 (:goal
  (and
      (contains shot394 cocktail222)
      (contains shot142 ingredient283)
)))
