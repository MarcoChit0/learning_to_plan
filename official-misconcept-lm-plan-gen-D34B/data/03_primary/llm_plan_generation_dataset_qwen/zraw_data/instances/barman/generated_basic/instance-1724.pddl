(define (problem prob)
 (:domain barman)
 (:objects 
      shaker496 - shaker
      left right - hand
      shot162 shot202 - shot
      ingredient345 ingredient380 ingredient212 ingredient36 - ingredient
      cocktail12 - cocktail
      dispenser133 dispenser139 dispenser269 dispenser62 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker496)
  (ontable shot162)
  (ontable shot202)
  (dispenses dispenser133 ingredient345)
  (dispenses dispenser139 ingredient380)
  (dispenses dispenser269 ingredient212)
  (dispenses dispenser62 ingredient36)
  (clean shaker496)
  (clean shot162)
  (clean shot202)
  (empty shaker496)
  (empty shot162)
  (empty shot202)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker496 l0)
  (shaker-level shaker496 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail12 ingredient345)
  (cocktail-part2 cocktail12 ingredient212)
)
 (:goal
  (and
      (contains shot162 cocktail12)
)))
