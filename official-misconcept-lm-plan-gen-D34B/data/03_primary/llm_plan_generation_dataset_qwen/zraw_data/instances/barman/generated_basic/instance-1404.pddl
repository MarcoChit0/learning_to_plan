(define (problem prob)
 (:domain barman)
 (:objects 
      shaker32 - shaker
      left right - hand
      shot275 - shot
      ingredient371 ingredient377 ingredient321 ingredient370 - ingredient
      cocktail40 - cocktail
      dispenser226 dispenser229 dispenser103 dispenser281 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker32)
  (ontable shot275)
  (dispenses dispenser226 ingredient371)
  (dispenses dispenser229 ingredient377)
  (dispenses dispenser103 ingredient321)
  (dispenses dispenser281 ingredient370)
  (clean shaker32)
  (clean shot275)
  (empty shaker32)
  (empty shot275)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker32 l0)
  (shaker-level shaker32 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail40 ingredient377)
  (cocktail-part2 cocktail40 ingredient321)
)
 (:goal
  (and
      (contains shot275 cocktail40)
)))
