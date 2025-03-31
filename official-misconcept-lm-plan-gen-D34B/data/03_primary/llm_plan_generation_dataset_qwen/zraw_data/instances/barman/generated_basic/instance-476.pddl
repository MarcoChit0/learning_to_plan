(define (problem prob)
 (:domain barman)
 (:objects 
      shaker299 - shaker
      left right - hand
      shot234 shot339 - shot
      ingredient498 ingredient105 ingredient34 ingredient14 - ingredient
      cocktail1 - cocktail
      dispenser65 dispenser215 dispenser14 dispenser402 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker299)
  (ontable shot234)
  (ontable shot339)
  (dispenses dispenser65 ingredient498)
  (dispenses dispenser215 ingredient105)
  (dispenses dispenser14 ingredient34)
  (dispenses dispenser402 ingredient14)
  (clean shaker299)
  (clean shot234)
  (clean shot339)
  (empty shaker299)
  (empty shot234)
  (empty shot339)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker299 l0)
  (shaker-level shaker299 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient105)
  (cocktail-part2 cocktail1 ingredient14)
)
 (:goal
  (and
      (contains shot234 cocktail1)
)))
