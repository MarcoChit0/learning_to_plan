(define (problem prob)
 (:domain barman)
 (:objects 
      shaker179 - shaker
      left right - hand
      shot443 shot420 shot486 - shot
      ingredient343 ingredient354 ingredient497 - ingredient
      cocktail188 - cocktail
      dispenser220 dispenser21 dispenser172 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker179)
  (ontable shot443)
  (ontable shot420)
  (ontable shot486)
  (dispenses dispenser220 ingredient343)
  (dispenses dispenser21 ingredient354)
  (dispenses dispenser172 ingredient497)
  (clean shaker179)
  (clean shot443)
  (clean shot420)
  (clean shot486)
  (empty shaker179)
  (empty shot443)
  (empty shot420)
  (empty shot486)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker179 l0)
  (shaker-level shaker179 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail188 ingredient354)
  (cocktail-part2 cocktail188 ingredient497)
)
 (:goal
  (and
      (contains shot443 cocktail188)
      (contains shot420 cocktail188)
)))
