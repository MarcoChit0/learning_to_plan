(define (problem prob)
 (:domain barman)
 (:objects 
      shaker87 - shaker
      left right - hand
      shot162 shot494 shot92 - shot
      ingredient67 ingredient374 ingredient120 - ingredient
      cocktail69 - cocktail
      dispenser392 dispenser96 dispenser417 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker87)
  (ontable shot162)
  (ontable shot494)
  (ontable shot92)
  (dispenses dispenser392 ingredient67)
  (dispenses dispenser96 ingredient374)
  (dispenses dispenser417 ingredient120)
  (clean shaker87)
  (clean shot162)
  (clean shot494)
  (clean shot92)
  (empty shaker87)
  (empty shot162)
  (empty shot494)
  (empty shot92)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker87 l0)
  (shaker-level shaker87 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail69 ingredient374)
  (cocktail-part2 cocktail69 ingredient120)
)
 (:goal
  (and
      (contains shot162 cocktail69)
      (contains shot494 cocktail69)
)))
