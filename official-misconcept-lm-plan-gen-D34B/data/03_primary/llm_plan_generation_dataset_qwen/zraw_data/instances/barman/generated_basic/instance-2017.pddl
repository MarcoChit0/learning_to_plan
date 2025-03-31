(define (problem prob)
 (:domain barman)
 (:objects 
      shaker320 - shaker
      left right - hand
      shot367 shot221 shot216 - shot
      ingredient101 ingredient236 ingredient478 - ingredient
      cocktail292 - cocktail
      dispenser293 dispenser426 dispenser333 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker320)
  (ontable shot367)
  (ontable shot221)
  (ontable shot216)
  (dispenses dispenser293 ingredient101)
  (dispenses dispenser426 ingredient236)
  (dispenses dispenser333 ingredient478)
  (clean shaker320)
  (clean shot367)
  (clean shot221)
  (clean shot216)
  (empty shaker320)
  (empty shot367)
  (empty shot221)
  (empty shot216)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker320 l0)
  (shaker-level shaker320 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail292 ingredient101)
  (cocktail-part2 cocktail292 ingredient478)
)
 (:goal
  (and
      (contains shot367 cocktail292)
      (contains shot221 cocktail292)
)))
