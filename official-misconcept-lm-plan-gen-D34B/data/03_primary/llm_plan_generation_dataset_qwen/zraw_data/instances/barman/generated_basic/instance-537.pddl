(define (problem prob)
 (:domain barman)
 (:objects 
      shaker358 - shaker
      left right - hand
      shot277 - shot
      ingredient217 ingredient444 ingredient265 - ingredient
      cocktail1 - cocktail
      dispenser488 dispenser135 dispenser158 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker358)
  (ontable shot277)
  (dispenses dispenser488 ingredient217)
  (dispenses dispenser135 ingredient444)
  (dispenses dispenser158 ingredient265)
  (clean shaker358)
  (clean shot277)
  (empty shaker358)
  (empty shot277)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker358 l0)
  (shaker-level shaker358 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient265)
  (cocktail-part2 cocktail1 ingredient444)
)
 (:goal
  (and
      (contains shot277 cocktail1)
)))
