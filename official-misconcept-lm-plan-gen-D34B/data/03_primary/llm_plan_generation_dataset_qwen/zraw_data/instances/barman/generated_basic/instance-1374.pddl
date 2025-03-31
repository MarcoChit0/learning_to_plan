(define (problem prob)
 (:domain barman)
 (:objects 
      shaker221 - shaker
      left right - hand
      shot113 shot5 shot319 - shot
      ingredient149 ingredient418 ingredient126 ingredient361 - ingredient
      cocktail345 - cocktail
      dispenser334 dispenser54 dispenser23 dispenser160 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker221)
  (ontable shot113)
  (ontable shot5)
  (ontable shot319)
  (dispenses dispenser334 ingredient149)
  (dispenses dispenser54 ingredient418)
  (dispenses dispenser23 ingredient126)
  (dispenses dispenser160 ingredient361)
  (clean shaker221)
  (clean shot113)
  (clean shot5)
  (clean shot319)
  (empty shaker221)
  (empty shot113)
  (empty shot5)
  (empty shot319)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker221 l0)
  (shaker-level shaker221 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail345 ingredient126)
  (cocktail-part2 cocktail345 ingredient149)
)
 (:goal
  (and
      (contains shot113 cocktail345)
      (contains shot5 cocktail345)
)))
