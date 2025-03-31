(define (problem prob)
 (:domain barman)
 (:objects 
      shaker468 - shaker
      left right - hand
      shot145 shot332 - shot
      ingredient462 ingredient409 ingredient16 ingredient77 - ingredient
      cocktail1 - cocktail
      dispenser443 dispenser201 dispenser494 dispenser456 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker468)
  (ontable shot145)
  (ontable shot332)
  (dispenses dispenser443 ingredient462)
  (dispenses dispenser201 ingredient409)
  (dispenses dispenser494 ingredient16)
  (dispenses dispenser456 ingredient77)
  (clean shaker468)
  (clean shot145)
  (clean shot332)
  (empty shaker468)
  (empty shot145)
  (empty shot332)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker468 l0)
  (shaker-level shaker468 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient77)
  (cocktail-part2 cocktail1 ingredient16)
)
 (:goal
  (and
      (contains shot145 cocktail1)
)))
