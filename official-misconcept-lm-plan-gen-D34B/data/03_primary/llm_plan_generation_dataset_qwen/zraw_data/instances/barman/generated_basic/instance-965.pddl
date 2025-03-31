(define (problem prob)
 (:domain barman)
 (:objects 
      shaker225 - shaker
      left right - hand
      shot239 - shot
      ingredient244 ingredient490 ingredient180 ingredient121 - ingredient
      cocktail1 - cocktail
      dispenser32 dispenser196 dispenser111 dispenser9 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker225)
  (ontable shot239)
  (dispenses dispenser32 ingredient244)
  (dispenses dispenser196 ingredient490)
  (dispenses dispenser111 ingredient180)
  (dispenses dispenser9 ingredient121)
  (clean shaker225)
  (clean shot239)
  (empty shaker225)
  (empty shot239)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker225 l0)
  (shaker-level shaker225 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient121)
  (cocktail-part2 cocktail1 ingredient490)
)
 (:goal
  (and
      (contains shot239 cocktail1)
)))
