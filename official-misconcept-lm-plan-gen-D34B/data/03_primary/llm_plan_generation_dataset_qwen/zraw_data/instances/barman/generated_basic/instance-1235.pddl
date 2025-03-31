(define (problem prob)
 (:domain barman)
 (:objects 
      shaker264 - shaker
      left right - hand
      shot373 shot25 shot237 - shot
      ingredient367 ingredient126 ingredient218 ingredient49 - ingredient
      cocktail437 - cocktail
      dispenser43 dispenser403 dispenser168 dispenser11 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker264)
  (ontable shot373)
  (ontable shot25)
  (ontable shot237)
  (dispenses dispenser43 ingredient367)
  (dispenses dispenser403 ingredient126)
  (dispenses dispenser168 ingredient218)
  (dispenses dispenser11 ingredient49)
  (clean shaker264)
  (clean shot373)
  (clean shot25)
  (clean shot237)
  (empty shaker264)
  (empty shot373)
  (empty shot25)
  (empty shot237)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker264 l0)
  (shaker-level shaker264 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail437 ingredient49)
  (cocktail-part2 cocktail437 ingredient367)
)
 (:goal
  (and
      (contains shot373 cocktail437)
      (contains shot25 cocktail437)
)))
