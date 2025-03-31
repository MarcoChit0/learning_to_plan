(define (problem prob)
 (:domain barman)
 (:objects 
      shaker206 - shaker
      left right - hand
      shot431 shot282 - shot
      ingredient23 ingredient424 ingredient111 - ingredient
      cocktail280 - cocktail
      dispenser422 dispenser100 dispenser227 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker206)
  (ontable shot431)
  (ontable shot282)
  (dispenses dispenser422 ingredient23)
  (dispenses dispenser100 ingredient424)
  (dispenses dispenser227 ingredient111)
  (clean shaker206)
  (clean shot431)
  (clean shot282)
  (empty shaker206)
  (empty shot431)
  (empty shot282)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker206 l0)
  (shaker-level shaker206 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail280 ingredient111)
  (cocktail-part2 cocktail280 ingredient424)
)
 (:goal
  (and
      (contains shot431 cocktail280)
)))
