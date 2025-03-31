(define (problem prob)
 (:domain barman)
 (:objects 
      shaker95 - shaker
      left right - hand
      shot318 shot367 - shot
      ingredient461 ingredient463 - ingredient
      cocktail436 - cocktail
      dispenser8 dispenser206 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker95)
  (ontable shot318)
  (ontable shot367)
  (dispenses dispenser8 ingredient461)
  (dispenses dispenser206 ingredient463)
  (clean shaker95)
  (clean shot318)
  (clean shot367)
  (empty shaker95)
  (empty shot318)
  (empty shot367)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker95 l0)
  (shaker-level shaker95 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail436 ingredient463)
  (cocktail-part2 cocktail436 ingredient461)
)
 (:goal
  (and
      (contains shot318 cocktail436)
)))
