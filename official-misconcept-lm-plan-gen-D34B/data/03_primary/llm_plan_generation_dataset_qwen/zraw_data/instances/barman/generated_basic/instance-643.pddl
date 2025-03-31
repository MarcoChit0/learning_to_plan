(define (problem prob)
 (:domain barman)
 (:objects 
      shaker436 - shaker
      left right - hand
      shot371 shot349 shot215 - shot
      ingredient460 ingredient123 ingredient255 - ingredient
      cocktail1 - cocktail
      dispenser14 dispenser3 dispenser327 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker436)
  (ontable shot371)
  (ontable shot349)
  (ontable shot215)
  (dispenses dispenser14 ingredient460)
  (dispenses dispenser3 ingredient123)
  (dispenses dispenser327 ingredient255)
  (clean shaker436)
  (clean shot371)
  (clean shot349)
  (clean shot215)
  (empty shaker436)
  (empty shot371)
  (empty shot349)
  (empty shot215)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker436 l0)
  (shaker-level shaker436 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient123)
  (cocktail-part2 cocktail1 ingredient460)
)
 (:goal
  (and
      (contains shot371 cocktail1)
      (contains shot349 ingredient460)
)))
