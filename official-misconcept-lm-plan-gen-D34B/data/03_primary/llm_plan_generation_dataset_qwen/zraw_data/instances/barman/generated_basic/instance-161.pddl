(define (problem prob)
 (:domain barman)
 (:objects 
      shaker53 - shaker
      left right - hand
      shot91 shot120 - shot
      ingredient352 ingredient360 ingredient254 - ingredient
      cocktail1 - cocktail
      dispenser400 dispenser250 dispenser385 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker53)
  (ontable shot91)
  (ontable shot120)
  (dispenses dispenser400 ingredient352)
  (dispenses dispenser250 ingredient360)
  (dispenses dispenser385 ingredient254)
  (clean shaker53)
  (clean shot91)
  (clean shot120)
  (empty shaker53)
  (empty shot91)
  (empty shot120)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker53 l0)
  (shaker-level shaker53 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient360)
  (cocktail-part2 cocktail1 ingredient352)
)
 (:goal
  (and
      (contains shot91 cocktail1)
)))
