(define (problem prob)
 (:domain barman)
 (:objects 
      shaker410 - shaker
      left right - hand
      shot407 shot170 - shot
      ingredient148 ingredient368 ingredient478 ingredient182 - ingredient
      cocktail78 - cocktail
      dispenser460 dispenser91 dispenser354 dispenser211 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker410)
  (ontable shot407)
  (ontable shot170)
  (dispenses dispenser460 ingredient148)
  (dispenses dispenser91 ingredient368)
  (dispenses dispenser354 ingredient478)
  (dispenses dispenser211 ingredient182)
  (clean shaker410)
  (clean shot407)
  (clean shot170)
  (empty shaker410)
  (empty shot407)
  (empty shot170)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker410 l0)
  (shaker-level shaker410 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail78 ingredient148)
  (cocktail-part2 cocktail78 ingredient368)
)
 (:goal
  (and
      (contains shot407 cocktail78)
)))
