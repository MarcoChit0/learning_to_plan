(define (problem prob)
 (:domain barman)
 (:objects 
      shaker281 - shaker
      left right - hand
      shot410 shot14 shot491 - shot
      ingredient16 ingredient376 ingredient65 ingredient416 - ingredient
      cocktail1 - cocktail
      dispenser455 dispenser406 dispenser295 dispenser165 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker281)
  (ontable shot410)
  (ontable shot14)
  (ontable shot491)
  (dispenses dispenser455 ingredient16)
  (dispenses dispenser406 ingredient376)
  (dispenses dispenser295 ingredient65)
  (dispenses dispenser165 ingredient416)
  (clean shaker281)
  (clean shot410)
  (clean shot14)
  (clean shot491)
  (empty shaker281)
  (empty shot410)
  (empty shot14)
  (empty shot491)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker281 l0)
  (shaker-level shaker281 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient376)
  (cocktail-part2 cocktail1 ingredient16)
)
 (:goal
  (and
      (contains shot410 cocktail1)
      (contains shot14 ingredient376)
)))
