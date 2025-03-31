(define (problem prob)
 (:domain barman)
 (:objects 
      shaker464 - shaker
      left right - hand
      shot363 shot446 shot349 - shot
      ingredient163 ingredient421 - ingredient
      cocktail331 - cocktail
      dispenser14 dispenser68 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker464)
  (ontable shot363)
  (ontable shot446)
  (ontable shot349)
  (dispenses dispenser14 ingredient163)
  (dispenses dispenser68 ingredient421)
  (clean shaker464)
  (clean shot363)
  (clean shot446)
  (clean shot349)
  (empty shaker464)
  (empty shot363)
  (empty shot446)
  (empty shot349)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker464 l0)
  (shaker-level shaker464 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail331 ingredient421)
  (cocktail-part2 cocktail331 ingredient163)
)
 (:goal
  (and
      (contains shot363 cocktail331)
      (contains shot446 cocktail331)
)))
