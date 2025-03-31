(define (problem prob)
 (:domain barman)
 (:objects 
      shaker298 - shaker
      left right - hand
      shot102 shot337 - shot
      ingredient385 ingredient31 ingredient171 ingredient393 - ingredient
      cocktail1 - cocktail
      dispenser411 dispenser385 dispenser52 dispenser311 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker298)
  (ontable shot102)
  (ontable shot337)
  (dispenses dispenser411 ingredient385)
  (dispenses dispenser385 ingredient31)
  (dispenses dispenser52 ingredient171)
  (dispenses dispenser311 ingredient393)
  (clean shaker298)
  (clean shot102)
  (clean shot337)
  (empty shaker298)
  (empty shot102)
  (empty shot337)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker298 l0)
  (shaker-level shaker298 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient393)
  (cocktail-part2 cocktail1 ingredient171)
)
 (:goal
  (and
      (contains shot102 cocktail1)
)))
