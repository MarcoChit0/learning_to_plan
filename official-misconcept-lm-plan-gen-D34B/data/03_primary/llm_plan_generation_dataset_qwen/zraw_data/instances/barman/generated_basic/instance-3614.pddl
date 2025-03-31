(define (problem prob)
 (:domain barman)
 (:objects 
      shaker260 - shaker
      left right - hand
      shot92 shot494 shot14 - shot
      ingredient121 ingredient116 ingredient299 - ingredient
      cocktail50 - cocktail
      dispenser281 dispenser498 dispenser495 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker260)
  (ontable shot92)
  (ontable shot494)
  (ontable shot14)
  (dispenses dispenser281 ingredient121)
  (dispenses dispenser498 ingredient116)
  (dispenses dispenser495 ingredient299)
  (clean shaker260)
  (clean shot92)
  (clean shot494)
  (clean shot14)
  (empty shaker260)
  (empty shot92)
  (empty shot494)
  (empty shot14)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker260 l0)
  (shaker-level shaker260 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail50 ingredient121)
  (cocktail-part2 cocktail50 ingredient299)
)
 (:goal
  (and
      (contains shot92 cocktail50)
      (contains shot494 ingredient116)
)))
