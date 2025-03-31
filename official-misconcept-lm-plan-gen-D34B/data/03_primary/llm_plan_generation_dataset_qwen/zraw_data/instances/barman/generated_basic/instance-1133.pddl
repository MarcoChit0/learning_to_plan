(define (problem prob)
 (:domain barman)
 (:objects 
      shaker478 - shaker
      left right - hand
      shot360 shot224 - shot
      ingredient157 ingredient355 ingredient162 - ingredient
      cocktail1 - cocktail
      dispenser209 dispenser407 dispenser239 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker478)
  (ontable shot360)
  (ontable shot224)
  (dispenses dispenser209 ingredient157)
  (dispenses dispenser407 ingredient355)
  (dispenses dispenser239 ingredient162)
  (clean shaker478)
  (clean shot360)
  (clean shot224)
  (empty shaker478)
  (empty shot360)
  (empty shot224)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker478 l0)
  (shaker-level shaker478 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient355)
  (cocktail-part2 cocktail1 ingredient157)
)
 (:goal
  (and
      (contains shot360 cocktail1)
)))
