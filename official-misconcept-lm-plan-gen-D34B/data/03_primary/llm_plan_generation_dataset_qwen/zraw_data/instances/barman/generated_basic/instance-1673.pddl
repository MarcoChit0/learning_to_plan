(define (problem prob)
 (:domain barman)
 (:objects 
      shaker235 - shaker
      left right - hand
      shot183 shot440 - shot
      ingredient250 ingredient198 ingredient10 ingredient230 - ingredient
      cocktail115 - cocktail
      dispenser242 dispenser164 dispenser222 dispenser27 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker235)
  (ontable shot183)
  (ontable shot440)
  (dispenses dispenser242 ingredient250)
  (dispenses dispenser164 ingredient198)
  (dispenses dispenser222 ingredient10)
  (dispenses dispenser27 ingredient230)
  (clean shaker235)
  (clean shot183)
  (clean shot440)
  (empty shaker235)
  (empty shot183)
  (empty shot440)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker235 l0)
  (shaker-level shaker235 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail115 ingredient198)
  (cocktail-part2 cocktail115 ingredient230)
)
 (:goal
  (and
      (contains shot183 cocktail115)
)))
