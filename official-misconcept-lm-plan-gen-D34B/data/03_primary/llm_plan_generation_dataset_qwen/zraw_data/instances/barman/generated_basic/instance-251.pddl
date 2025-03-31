(define (problem prob)
 (:domain barman)
 (:objects 
      shaker213 - shaker
      left right - hand
      shot291 shot49 shot91 - shot
      ingredient429 ingredient327 ingredient30 - ingredient
      cocktail1 - cocktail
      dispenser277 dispenser253 dispenser216 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker213)
  (ontable shot291)
  (ontable shot49)
  (ontable shot91)
  (dispenses dispenser277 ingredient429)
  (dispenses dispenser253 ingredient327)
  (dispenses dispenser216 ingredient30)
  (clean shaker213)
  (clean shot291)
  (clean shot49)
  (clean shot91)
  (empty shaker213)
  (empty shot291)
  (empty shot49)
  (empty shot91)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker213 l0)
  (shaker-level shaker213 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient30)
  (cocktail-part2 cocktail1 ingredient327)
)
 (:goal
  (and
      (contains shot291 cocktail1)
      (contains shot49 cocktail1)
)))
