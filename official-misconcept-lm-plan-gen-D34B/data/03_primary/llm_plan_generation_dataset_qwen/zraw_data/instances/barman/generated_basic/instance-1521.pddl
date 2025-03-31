(define (problem prob)
 (:domain barman)
 (:objects 
      shaker360 - shaker
      left right - hand
      shot264 shot166 - shot
      ingredient367 ingredient199 ingredient164 ingredient215 - ingredient
      cocktail485 - cocktail
      dispenser155 dispenser104 dispenser256 dispenser287 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker360)
  (ontable shot264)
  (ontable shot166)
  (dispenses dispenser155 ingredient367)
  (dispenses dispenser104 ingredient199)
  (dispenses dispenser256 ingredient164)
  (dispenses dispenser287 ingredient215)
  (clean shaker360)
  (clean shot264)
  (clean shot166)
  (empty shaker360)
  (empty shot264)
  (empty shot166)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker360 l0)
  (shaker-level shaker360 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail485 ingredient215)
  (cocktail-part2 cocktail485 ingredient367)
)
 (:goal
  (and
      (contains shot264 cocktail485)
)))
