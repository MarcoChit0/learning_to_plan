(define (problem prob)
 (:domain barman)
 (:objects 
      shaker326 - shaker
      left right - hand
      shot191 shot226 - shot
      ingredient306 ingredient217 ingredient466 ingredient155 - ingredient
      cocktail126 - cocktail
      dispenser488 dispenser273 dispenser392 dispenser467 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker326)
  (ontable shot191)
  (ontable shot226)
  (dispenses dispenser488 ingredient306)
  (dispenses dispenser273 ingredient217)
  (dispenses dispenser392 ingredient466)
  (dispenses dispenser467 ingredient155)
  (clean shaker326)
  (clean shot191)
  (clean shot226)
  (empty shaker326)
  (empty shot191)
  (empty shot226)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker326 l0)
  (shaker-level shaker326 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail126 ingredient155)
  (cocktail-part2 cocktail126 ingredient217)
)
 (:goal
  (and
      (contains shot191 cocktail126)
)))
