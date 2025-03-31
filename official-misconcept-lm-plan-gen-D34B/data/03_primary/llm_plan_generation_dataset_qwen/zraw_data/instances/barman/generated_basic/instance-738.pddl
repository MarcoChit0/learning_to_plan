(define (problem prob)
 (:domain barman)
 (:objects 
      shaker38 - shaker
      left right - hand
      shot203 shot160 - shot
      ingredient174 ingredient223 ingredient213 - ingredient
      cocktail1 - cocktail
      dispenser488 dispenser421 dispenser433 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker38)
  (ontable shot203)
  (ontable shot160)
  (dispenses dispenser488 ingredient174)
  (dispenses dispenser421 ingredient223)
  (dispenses dispenser433 ingredient213)
  (clean shaker38)
  (clean shot203)
  (clean shot160)
  (empty shaker38)
  (empty shot203)
  (empty shot160)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker38 l0)
  (shaker-level shaker38 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient174)
  (cocktail-part2 cocktail1 ingredient223)
)
 (:goal
  (and
      (contains shot203 cocktail1)
)))
