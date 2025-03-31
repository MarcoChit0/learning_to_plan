(define (problem prob)
 (:domain barman)
 (:objects 
      shaker61 - shaker
      left right - hand
      shot419 shot276 shot91 - shot
      ingredient355 ingredient184 - ingredient
      cocktail384 - cocktail
      dispenser38 dispenser219 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker61)
  (ontable shot419)
  (ontable shot276)
  (ontable shot91)
  (dispenses dispenser38 ingredient355)
  (dispenses dispenser219 ingredient184)
  (clean shaker61)
  (clean shot419)
  (clean shot276)
  (clean shot91)
  (empty shaker61)
  (empty shot419)
  (empty shot276)
  (empty shot91)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker61 l0)
  (shaker-level shaker61 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail384 ingredient355)
  (cocktail-part2 cocktail384 ingredient184)
)
 (:goal
  (and
      (contains shot419 cocktail384)
      (contains shot276 ingredient184)
)))
