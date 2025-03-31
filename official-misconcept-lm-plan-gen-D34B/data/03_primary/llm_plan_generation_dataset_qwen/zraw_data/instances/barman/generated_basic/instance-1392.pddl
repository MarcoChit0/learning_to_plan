(define (problem prob)
 (:domain barman)
 (:objects 
      shaker270 - shaker
      left right - hand
      shot464 shot190 shot0 - shot
      ingredient452 ingredient42 ingredient456 ingredient341 - ingredient
      cocktail183 - cocktail
      dispenser377 dispenser180 dispenser316 dispenser314 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker270)
  (ontable shot464)
  (ontable shot190)
  (ontable shot0)
  (dispenses dispenser377 ingredient452)
  (dispenses dispenser180 ingredient42)
  (dispenses dispenser316 ingredient456)
  (dispenses dispenser314 ingredient341)
  (clean shaker270)
  (clean shot464)
  (clean shot190)
  (clean shot0)
  (empty shaker270)
  (empty shot464)
  (empty shot190)
  (empty shot0)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker270 l0)
  (shaker-level shaker270 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail183 ingredient456)
  (cocktail-part2 cocktail183 ingredient42)
)
 (:goal
  (and
      (contains shot464 cocktail183)
      (contains shot190 cocktail183)
)))
