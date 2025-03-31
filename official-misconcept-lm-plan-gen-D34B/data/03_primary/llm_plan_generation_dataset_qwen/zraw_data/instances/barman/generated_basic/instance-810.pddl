(define (problem prob)
 (:domain barman)
 (:objects 
      shaker124 - shaker
      left right - hand
      shot346 shot2 - shot
      ingredient345 ingredient427 ingredient20 ingredient160 - ingredient
      cocktail1 - cocktail
      dispenser481 dispenser206 dispenser273 dispenser126 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker124)
  (ontable shot346)
  (ontable shot2)
  (dispenses dispenser481 ingredient345)
  (dispenses dispenser206 ingredient427)
  (dispenses dispenser273 ingredient20)
  (dispenses dispenser126 ingredient160)
  (clean shaker124)
  (clean shot346)
  (clean shot2)
  (empty shaker124)
  (empty shot346)
  (empty shot2)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker124 l0)
  (shaker-level shaker124 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient20)
  (cocktail-part2 cocktail1 ingredient160)
)
 (:goal
  (and
      (contains shot346 cocktail1)
)))
