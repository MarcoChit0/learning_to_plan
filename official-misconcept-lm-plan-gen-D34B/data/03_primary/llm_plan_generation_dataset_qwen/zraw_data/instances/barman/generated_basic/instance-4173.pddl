(define (problem prob)
 (:domain barman)
 (:objects 
      shaker316 - shaker
      left right - hand
      shot492 shot379 - shot
      ingredient478 ingredient135 - ingredient
      cocktail56 - cocktail
      dispenser101 dispenser203 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker316)
  (ontable shot492)
  (ontable shot379)
  (dispenses dispenser101 ingredient478)
  (dispenses dispenser203 ingredient135)
  (clean shaker316)
  (clean shot492)
  (clean shot379)
  (empty shaker316)
  (empty shot492)
  (empty shot379)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker316 l0)
  (shaker-level shaker316 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail56 ingredient135)
  (cocktail-part2 cocktail56 ingredient478)
)
 (:goal
  (and
      (contains shot492 cocktail56)
)))
