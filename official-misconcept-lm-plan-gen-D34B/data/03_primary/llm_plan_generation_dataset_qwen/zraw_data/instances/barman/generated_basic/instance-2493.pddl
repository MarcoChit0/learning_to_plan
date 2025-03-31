(define (problem prob)
 (:domain barman)
 (:objects 
      shaker262 - shaker
      left right - hand
      shot207 shot368 shot367 - shot
      ingredient260 ingredient459 ingredient342 - ingredient
      cocktail184 - cocktail
      dispenser210 dispenser346 dispenser456 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker262)
  (ontable shot207)
  (ontable shot368)
  (ontable shot367)
  (dispenses dispenser210 ingredient260)
  (dispenses dispenser346 ingredient459)
  (dispenses dispenser456 ingredient342)
  (clean shaker262)
  (clean shot207)
  (clean shot368)
  (clean shot367)
  (empty shaker262)
  (empty shot207)
  (empty shot368)
  (empty shot367)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker262 l0)
  (shaker-level shaker262 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail184 ingredient260)
  (cocktail-part2 cocktail184 ingredient342)
)
 (:goal
  (and
      (contains shot207 cocktail184)
      (contains shot368 ingredient260)
)))
