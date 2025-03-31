(define (problem prob)
 (:domain barman)
 (:objects 
      shaker63 - shaker
      left right - hand
      shot367 shot245 - shot
      ingredient367 ingredient174 - ingredient
      cocktail57 - cocktail
      dispenser17 dispenser19 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker63)
  (ontable shot367)
  (ontable shot245)
  (dispenses dispenser17 ingredient367)
  (dispenses dispenser19 ingredient174)
  (clean shaker63)
  (clean shot367)
  (clean shot245)
  (empty shaker63)
  (empty shot367)
  (empty shot245)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker63 l0)
  (shaker-level shaker63 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail57 ingredient174)
  (cocktail-part2 cocktail57 ingredient367)
)
 (:goal
  (and
      (contains shot367 cocktail57)
)))
