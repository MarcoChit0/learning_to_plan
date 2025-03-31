(define (problem prob)
 (:domain barman)
 (:objects 
      shaker428 - shaker
      left right - hand
      shot478 shot367 shot337 - shot
      ingredient147 ingredient329 ingredient40 - ingredient
      cocktail1 - cocktail
      dispenser290 dispenser168 dispenser122 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker428)
  (ontable shot478)
  (ontable shot367)
  (ontable shot337)
  (dispenses dispenser290 ingredient147)
  (dispenses dispenser168 ingredient329)
  (dispenses dispenser122 ingredient40)
  (clean shaker428)
  (clean shot478)
  (clean shot367)
  (clean shot337)
  (empty shaker428)
  (empty shot478)
  (empty shot367)
  (empty shot337)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker428 l0)
  (shaker-level shaker428 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient329)
  (cocktail-part2 cocktail1 ingredient147)
)
 (:goal
  (and
      (contains shot478 cocktail1)
      (contains shot367 cocktail1)
)))
