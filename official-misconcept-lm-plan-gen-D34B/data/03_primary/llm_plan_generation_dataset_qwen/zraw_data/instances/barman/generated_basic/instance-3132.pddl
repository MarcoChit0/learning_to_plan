(define (problem prob)
 (:domain barman)
 (:objects 
      shaker167 - shaker
      left right - hand
      shot28 shot301 shot367 - shot
      ingredient299 ingredient149 - ingredient
      cocktail121 - cocktail
      dispenser23 dispenser215 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker167)
  (ontable shot28)
  (ontable shot301)
  (ontable shot367)
  (dispenses dispenser23 ingredient299)
  (dispenses dispenser215 ingredient149)
  (clean shaker167)
  (clean shot28)
  (clean shot301)
  (clean shot367)
  (empty shaker167)
  (empty shot28)
  (empty shot301)
  (empty shot367)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker167 l0)
  (shaker-level shaker167 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail121 ingredient149)
  (cocktail-part2 cocktail121 ingredient299)
)
 (:goal
  (and
      (contains shot28 cocktail121)
      (contains shot301 ingredient299)
)))
