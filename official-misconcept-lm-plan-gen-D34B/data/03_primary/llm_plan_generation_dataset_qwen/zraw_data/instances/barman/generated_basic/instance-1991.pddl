(define (problem prob)
 (:domain barman)
 (:objects 
      shaker242 - shaker
      left right - hand
      shot15 shot191 - shot
      ingredient154 ingredient228 ingredient100 - ingredient
      cocktail460 - cocktail
      dispenser454 dispenser392 dispenser64 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker242)
  (ontable shot15)
  (ontable shot191)
  (dispenses dispenser454 ingredient154)
  (dispenses dispenser392 ingredient228)
  (dispenses dispenser64 ingredient100)
  (clean shaker242)
  (clean shot15)
  (clean shot191)
  (empty shaker242)
  (empty shot15)
  (empty shot191)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker242 l0)
  (shaker-level shaker242 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail460 ingredient100)
  (cocktail-part2 cocktail460 ingredient228)
)
 (:goal
  (and
      (contains shot15 cocktail460)
)))
