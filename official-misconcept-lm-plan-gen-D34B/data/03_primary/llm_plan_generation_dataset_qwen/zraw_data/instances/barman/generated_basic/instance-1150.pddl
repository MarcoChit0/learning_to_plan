(define (problem prob)
 (:domain barman)
 (:objects 
      shaker157 - shaker
      left right - hand
      shot368 shot239 - shot
      ingredient422 ingredient324 ingredient203 ingredient396 - ingredient
      cocktail1 - cocktail
      dispenser399 dispenser406 dispenser386 dispenser298 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker157)
  (ontable shot368)
  (ontable shot239)
  (dispenses dispenser399 ingredient422)
  (dispenses dispenser406 ingredient324)
  (dispenses dispenser386 ingredient203)
  (dispenses dispenser298 ingredient396)
  (clean shaker157)
  (clean shot368)
  (clean shot239)
  (empty shaker157)
  (empty shot368)
  (empty shot239)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker157 l0)
  (shaker-level shaker157 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient203)
  (cocktail-part2 cocktail1 ingredient422)
)
 (:goal
  (and
      (contains shot368 cocktail1)
)))
