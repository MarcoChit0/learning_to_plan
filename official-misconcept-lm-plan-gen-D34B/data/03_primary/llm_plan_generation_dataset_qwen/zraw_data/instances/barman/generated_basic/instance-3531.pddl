(define (problem prob)
 (:domain barman)
 (:objects 
      shaker447 - shaker
      left right - hand
      shot166 shot80 - shot
      ingredient34 ingredient337 ingredient364 ingredient264 - ingredient
      cocktail225 - cocktail
      dispenser97 dispenser365 dispenser357 dispenser41 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker447)
  (ontable shot166)
  (ontable shot80)
  (dispenses dispenser97 ingredient34)
  (dispenses dispenser365 ingredient337)
  (dispenses dispenser357 ingredient364)
  (dispenses dispenser41 ingredient264)
  (clean shaker447)
  (clean shot166)
  (clean shot80)
  (empty shaker447)
  (empty shot166)
  (empty shot80)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker447 l0)
  (shaker-level shaker447 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail225 ingredient34)
  (cocktail-part2 cocktail225 ingredient364)
)
 (:goal
  (and
      (contains shot166 cocktail225)
)))
