(define (problem prob)
 (:domain barman)
 (:objects 
      shaker154 - shaker
      left right - hand
      shot103 shot290 - shot
      ingredient394 ingredient284 ingredient321 ingredient451 - ingredient
      cocktail412 - cocktail
      dispenser49 dispenser203 dispenser332 dispenser475 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker154)
  (ontable shot103)
  (ontable shot290)
  (dispenses dispenser49 ingredient394)
  (dispenses dispenser203 ingredient284)
  (dispenses dispenser332 ingredient321)
  (dispenses dispenser475 ingredient451)
  (clean shaker154)
  (clean shot103)
  (clean shot290)
  (empty shaker154)
  (empty shot103)
  (empty shot290)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker154 l0)
  (shaker-level shaker154 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail412 ingredient284)
  (cocktail-part2 cocktail412 ingredient451)
)
 (:goal
  (and
      (contains shot103 cocktail412)
)))
