(define (problem prob)
 (:domain barman)
 (:objects 
      shaker189 - shaker
      left right - hand
      shot252 shot136 shot70 - shot
      ingredient349 ingredient229 ingredient366 - ingredient
      cocktail407 - cocktail
      dispenser453 dispenser358 dispenser194 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker189)
  (ontable shot252)
  (ontable shot136)
  (ontable shot70)
  (dispenses dispenser453 ingredient349)
  (dispenses dispenser358 ingredient229)
  (dispenses dispenser194 ingredient366)
  (clean shaker189)
  (clean shot252)
  (clean shot136)
  (clean shot70)
  (empty shaker189)
  (empty shot252)
  (empty shot136)
  (empty shot70)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker189 l0)
  (shaker-level shaker189 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail407 ingredient349)
  (cocktail-part2 cocktail407 ingredient366)
)
 (:goal
  (and
      (contains shot252 cocktail407)
      (contains shot136 ingredient349)
)))
