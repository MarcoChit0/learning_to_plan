(define (problem prob)
 (:domain barman)
 (:objects 
      shaker172 - shaker
      left right - hand
      shot296 shot297 - shot
      ingredient396 ingredient99 ingredient226 - ingredient
      cocktail246 - cocktail
      dispenser423 dispenser108 dispenser74 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker172)
  (ontable shot296)
  (ontable shot297)
  (dispenses dispenser423 ingredient396)
  (dispenses dispenser108 ingredient99)
  (dispenses dispenser74 ingredient226)
  (clean shaker172)
  (clean shot296)
  (clean shot297)
  (empty shaker172)
  (empty shot296)
  (empty shot297)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker172 l0)
  (shaker-level shaker172 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail246 ingredient99)
  (cocktail-part2 cocktail246 ingredient226)
)
 (:goal
  (and
      (contains shot296 cocktail246)
)))
