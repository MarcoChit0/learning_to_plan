(define (problem prob)
 (:domain barman)
 (:objects 
      shaker345 - shaker
      left right - hand
      shot139 shot417 shot316 - shot
      ingredient334 ingredient166 ingredient5 ingredient433 - ingredient
      cocktail485 - cocktail
      dispenser154 dispenser43 dispenser30 dispenser40 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker345)
  (ontable shot139)
  (ontable shot417)
  (ontable shot316)
  (dispenses dispenser154 ingredient334)
  (dispenses dispenser43 ingredient166)
  (dispenses dispenser30 ingredient5)
  (dispenses dispenser40 ingredient433)
  (clean shaker345)
  (clean shot139)
  (clean shot417)
  (clean shot316)
  (empty shaker345)
  (empty shot139)
  (empty shot417)
  (empty shot316)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker345 l0)
  (shaker-level shaker345 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail485 ingredient433)
  (cocktail-part2 cocktail485 ingredient334)
)
 (:goal
  (and
      (contains shot139 cocktail485)
      (contains shot417 cocktail485)
)))
