(define (problem prob)
 (:domain barman)
 (:objects 
      shaker32 - shaker
      left right - hand
      shot367 - shot
      ingredient22 ingredient202 ingredient435 - ingredient
      cocktail237 - cocktail
      dispenser491 dispenser158 dispenser493 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker32)
  (ontable shot367)
  (dispenses dispenser491 ingredient22)
  (dispenses dispenser158 ingredient202)
  (dispenses dispenser493 ingredient435)
  (clean shaker32)
  (clean shot367)
  (empty shaker32)
  (empty shot367)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker32 l0)
  (shaker-level shaker32 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail237 ingredient22)
  (cocktail-part2 cocktail237 ingredient202)
)
 (:goal
  (and
      (contains shot367 cocktail237)
)))
