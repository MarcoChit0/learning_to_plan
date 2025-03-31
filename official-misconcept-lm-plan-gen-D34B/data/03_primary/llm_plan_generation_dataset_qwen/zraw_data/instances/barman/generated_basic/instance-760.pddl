(define (problem prob)
 (:domain barman)
 (:objects 
      shaker157 - shaker
      left right - hand
      shot6 shot36 shot250 - shot
      ingredient149 ingredient368 ingredient180 - ingredient
      cocktail1 - cocktail
      dispenser293 dispenser466 dispenser95 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker157)
  (ontable shot6)
  (ontable shot36)
  (ontable shot250)
  (dispenses dispenser293 ingredient149)
  (dispenses dispenser466 ingredient368)
  (dispenses dispenser95 ingredient180)
  (clean shaker157)
  (clean shot6)
  (clean shot36)
  (clean shot250)
  (empty shaker157)
  (empty shot6)
  (empty shot36)
  (empty shot250)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker157 l0)
  (shaker-level shaker157 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient149)
  (cocktail-part2 cocktail1 ingredient180)
)
 (:goal
  (and
      (contains shot6 cocktail1)
      (contains shot36 cocktail1)
)))
