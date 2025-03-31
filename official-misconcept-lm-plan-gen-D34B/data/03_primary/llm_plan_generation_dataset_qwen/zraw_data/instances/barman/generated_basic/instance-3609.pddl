(define (problem prob)
 (:domain barman)
 (:objects 
      shaker168 - shaker
      left right - hand
      shot396 shot201 shot246 - shot
      ingredient27 ingredient298 ingredient419 - ingredient
      cocktail230 - cocktail
      dispenser175 dispenser21 dispenser458 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker168)
  (ontable shot396)
  (ontable shot201)
  (ontable shot246)
  (dispenses dispenser175 ingredient27)
  (dispenses dispenser21 ingredient298)
  (dispenses dispenser458 ingredient419)
  (clean shaker168)
  (clean shot396)
  (clean shot201)
  (clean shot246)
  (empty shaker168)
  (empty shot396)
  (empty shot201)
  (empty shot246)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker168 l0)
  (shaker-level shaker168 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail230 ingredient27)
  (cocktail-part2 cocktail230 ingredient419)
)
 (:goal
  (and
      (contains shot396 cocktail230)
      (contains shot201 cocktail230)
)))
