(define (problem prob)
 (:domain barman)
 (:objects 
      shaker264 - shaker
      left right - hand
      shot440 shot443 - shot
      ingredient114 ingredient195 - ingredient
      cocktail1 - cocktail
      dispenser100 dispenser173 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker264)
  (ontable shot440)
  (ontable shot443)
  (dispenses dispenser100 ingredient114)
  (dispenses dispenser173 ingredient195)
  (clean shaker264)
  (clean shot440)
  (clean shot443)
  (empty shaker264)
  (empty shot440)
  (empty shot443)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker264 l0)
  (shaker-level shaker264 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient195)
  (cocktail-part2 cocktail1 ingredient114)
)
 (:goal
  (and
      (contains shot440 cocktail1)
)))
