(define (problem prob)
 (:domain barman)
 (:objects 
      shaker139 - shaker
      left right - hand
      shot456 shot358 shot278 - shot
      ingredient189 ingredient436 ingredient33 ingredient308 - ingredient
      cocktail1 - cocktail
      dispenser286 dispenser389 dispenser260 dispenser299 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker139)
  (ontable shot456)
  (ontable shot358)
  (ontable shot278)
  (dispenses dispenser286 ingredient189)
  (dispenses dispenser389 ingredient436)
  (dispenses dispenser260 ingredient33)
  (dispenses dispenser299 ingredient308)
  (clean shaker139)
  (clean shot456)
  (clean shot358)
  (clean shot278)
  (empty shaker139)
  (empty shot456)
  (empty shot358)
  (empty shot278)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker139 l0)
  (shaker-level shaker139 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient33)
  (cocktail-part2 cocktail1 ingredient189)
)
 (:goal
  (and
      (contains shot456 cocktail1)
      (contains shot358 cocktail1)
)))
