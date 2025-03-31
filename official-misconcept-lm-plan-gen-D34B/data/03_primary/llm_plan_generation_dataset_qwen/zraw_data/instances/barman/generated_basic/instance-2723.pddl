(define (problem prob)
 (:domain barman)
 (:objects 
      shaker404 - shaker
      left right - hand
      shot44 shot371 shot38 - shot
      ingredient316 ingredient358 ingredient6 - ingredient
      cocktail385 - cocktail
      dispenser23 dispenser14 dispenser45 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker404)
  (ontable shot44)
  (ontable shot371)
  (ontable shot38)
  (dispenses dispenser23 ingredient316)
  (dispenses dispenser14 ingredient358)
  (dispenses dispenser45 ingredient6)
  (clean shaker404)
  (clean shot44)
  (clean shot371)
  (clean shot38)
  (empty shaker404)
  (empty shot44)
  (empty shot371)
  (empty shot38)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker404 l0)
  (shaker-level shaker404 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail385 ingredient6)
  (cocktail-part2 cocktail385 ingredient358)
)
 (:goal
  (and
      (contains shot44 cocktail385)
      (contains shot371 ingredient316)
)))
